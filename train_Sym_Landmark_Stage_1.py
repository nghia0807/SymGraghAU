import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
from datetime import datetime

from model.SymStageLandmark1 import SymStageLandmark1
from dataset import DISFA_Landmark
from utils import *
from conf import get_config, set_logger, set_outdir, set_env


# ── AU-Expression mapping matrix ─────────────────────────────────────────────
M_AE_np = np.load(r"matrixMAE\M_AE_DISFA.npy")
M_AE    = torch.from_numpy(M_AE_np).float()


def to_stage2_compatible_state_dict(state_dict):
    """Map SymStageLandmark1 keys sang định dạng MEFARGStage1 cho Stage 2/3."""
    from collections import OrderedDict

    out = OrderedDict()
    for k, v in state_dict.items():
        key = k[7:] if k.startswith('module.') else k

        # Stage 2 không có geometric branch, bỏ các key này.
        if (key.startswith('landmark_gcn.') or
                key.startswith('au_pool.') or
                key.startswith('fusion.')):
            continue

        # Map AU branch naming: au_extractors/classifiers -> au_head.*
        if key.startswith('au_extractors.'):
            key = 'au_head.' + key[len('au_'):]
        elif key.startswith('au_classifiers.'):
            key = 'au_head.' + key[len('au_'):]

        out[key] = v

    return out


def get_dataloader(conf):
    print('==> Preparing data...')
    trainset = DISFA_Landmark(
        root_path=conf.dataset_path,
        train=True,
        fold=conf.fold,
        transform=image_train(crop_size=conf.crop_size),
        crop_size=conf.crop_size,
        landmark_zeros_on_miss=True,   # an toàn khi .npy chưa extract xong
    )
    train_loader = DataLoader(trainset, batch_size=conf.batch_size,
                              shuffle=True, num_workers=conf.num_workers)

    valset = DISFA_Landmark(
        root_path=conf.dataset_path,
        train=False,
        fold=conf.fold,
        transform=image_test(crop_size=conf.crop_size),
        landmark_zeros_on_miss=True,
    )
    val_loader = DataLoader(valset, batch_size=conf.batch_size,
                            shuffle=False, num_workers=conf.num_workers)

    return train_loader, val_loader, len(trainset), len(valset)


def au_to_expr_pseudo(Y_a: torch.Tensor,
                      M_AE: torch.Tensor,
                      neutral_index: int = 6) -> torch.Tensor:
    """Tạo pseudo-label expression từ AU labels (giống train_Sym_Stage_1)."""
    Y_a_float = Y_a.float()
    scores    = Y_a_float @ M_AE              # (B, N_e)
    ke        = scores.argmax(dim=1)          # (B,)
    ke[Y_a_float.sum(dim=1) == 0] = neutral_index
    Y_e = torch.zeros(Y_a.size(0), M_AE.size(1),
                      device=Y_a.device, dtype=torch.float32)
    Y_e.scatter_(1, ke.unsqueeze(1), 1.0)
    return Y_e


# ── Train one epoch ───────────────────────────────────────────────────────────

def train(conf, net, train_loader, optimizer, epoch, criterion, criterion_Em):
    losses = AverageMeter()
    net.train()
    loader_len = len(train_loader)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for batch_idx, (inputs, landmarks, targets) in enumerate(pbar):
        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate,
                             batch_idx, loader_len)

        targets = targets.float()
        if torch.cuda.is_available():
            inputs    = inputs.cuda()
            landmarks = landmarks.cuda()
            targets   = targets.cuda()

        # Pseudo expression labels
        targets_Emo = au_to_expr_pseudo(targets, M_AE)
        if torch.cuda.is_available():
            targets_Emo = targets_Emo.cuda()

        optimizer.zero_grad()

        # Forward – truyền cả landmarks
        V_a, V_e, outputs_AU, outputs_Emo = net(inputs, landmarks)

        # Loss
        L_wa = criterion(outputs_AU, targets)          # Weighted AU loss
        L_we = criterion_Em(outputs_Emo, targets_Emo)  # Expression BCE loss
        loss = L_wa + L_we

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        pbar.set_postfix({
            'L_wa': f"{L_wa.item():.4f}",
            'L_we': f"{L_we.item():.4f}",
            'L_jf': f"{loss.item():.4f}",
        })

    return losses.avg


# ── Validation ────────────────────────────────────────────────────────────────

def val(net, val_loader, criterion):
    losses         = AverageMeter()
    statistics_list = None
    net.eval()
    pbar = tqdm(val_loader, desc="Val")

    for batch_idx, (inputs, landmarks, targets) in enumerate(pbar):
        with torch.no_grad():
            targets = targets.float()
            if torch.cuda.is_available():
                inputs    = inputs.cuda()
                landmarks = landmarks.cuda()
                targets   = targets.cuda()

            V_a, V_e, outputs_AU, outputs_Emo = net(inputs, landmarks)

            loss = criterion(outputs_AU, targets)
            losses.update(loss.item(), inputs.size(0))

            update_list    = statistics(outputs_AU, targets.detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)

            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list           = calc_acc(statistics_list)
    return losses.avg, mean_f1_score, f1_score_list, mean_acc, acc_list


# ── Main ──────────────────────────────────────────────────────────────────────

def main(conf):
    if conf.dataset != 'DISFA':
        raise ValueError('train_Sym_Landmark_Stage_1 hiện chỉ hỗ trợ DISFA')

    start_epoch = 0

    # Data
    train_loader, val_loader, train_data_num, val_data_num = get_dataloader(conf)

    # Class weights cho Weighted Asymmetric Loss
    weight_path  = os.path.join(conf.dataset_path, 'list',
                                f'DISFA_train_weight_fold{conf.fold}.txt')
    train_weight = torch.from_numpy(np.loadtxt(weight_path)).float()
    print(f"[WAL] weight = {train_weight.tolist()}")

    logging.info(f"Fold: [{conf.fold} | {conf.N_fold}  val_data_num: {val_data_num}]")

    # Model
    net = SymStageLandmark1(
        num_aus=conf.num_classes,
        num_expr=7,
        backbone=conf.arc,
        gcn_hidden=(64, 128),
        use_landmark=True,
    )

    # Resume
    if conf.resume != '':
        logging.info(f"Resume from: {conf.resume}")
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net          = nn.DataParallel(net).cuda()
        train_weight = train_weight.cuda()
        global M_AE
        M_AE = M_AE.cuda()

    criterion    = WeightedAsymmetricLoss(weight=train_weight)
    criterion_Em = ExpressionBCELoss()
    optimizer    = optim.AdamW(net.parameters(), betas=(0.9, 0.999),
                               lr=conf.learning_rate, weight_decay=conf.weight_decay)

    print(f"Learning rate : {conf.learning_rate}")
    print(f"Train samples : {train_data_num:,}   Val samples: {val_data_num:,}")

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Model params  : {total_params:,}")

    start_time = datetime.now()
    print(f"Start time    : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    best_f1    = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, conf.epochs):
        logging.info(f"[CFG] arc={conf.arc} bs={conf.batch_size} "
                     f"lr0={conf.learning_rate} wd={conf.weight_decay}")
        lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch: [{epoch + 1} | {conf.epochs}  LR: {lr:.2e}]")

        # ── Train ──
        train_loss = train(conf, net, train_loader, optimizer, epoch,
                           criterion, criterion_Em)

        # ── Val ──
        val_loss, val_mean_f1, val_f1_list, val_mean_acc, val_acc_list = \
            val(net, val_loader, criterion)

        # ── Log ──
        print(f"Epoch {epoch+1:>3}/{conf.epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"F1={100.*val_mean_f1:.2f}%  "
              f"Acc={100.*val_mean_acc:.2f}%")

        # F1 per AU
        au_names = ['AU1','AU2','AU4','AU6','AU9','AU12','AU25','AU26']
        f1_strs  = '  '.join(f"{n}:{100.*v:.1f}" for n, v in zip(au_names, val_f1_list))
        print(f"  F1 per AU: {f1_strs}")

        # Alpha gate (chỉ in mỗi 4 epoch)
        if (epoch + 1) % 4 == 0:
            raw_net = net.module if hasattr(net, 'module') else net
            alpha   = raw_net.get_alpha()
            if alpha is not None:
                a_strs = '  '.join(f"{n}:{v:.3f}" for n, v in zip(au_names, alpha))
                print(f"  Alpha gate: {a_strs}")

        # Best model
        if val_mean_f1 > best_f1:
            best_f1    = val_mean_f1
            best_epoch = epoch + 1
            best_ckpt  = os.path.join(conf['outdir'], f'best_model_fold{conf.fold}.pth')
            raw_state_dict = net.state_dict()
            torch.save({
                'epoch':      epoch,
                'state_dict': raw_state_dict,
                'optimizer':  optimizer.state_dict(),
                'val_f1':     val_mean_f1,
            }, best_ckpt)
            print(f"  ** Best F1 updated: {100.*best_f1:.2f}% → saved to {best_ckpt}")

            best_ckpt_stage2 = os.path.join(
                conf['outdir'],
                f'best_model_stage2_ready_fold{conf.fold}.pth'
            )
            torch.save({
                'epoch':      epoch,
                'state_dict': to_stage2_compatible_state_dict(raw_state_dict),
                'val_f1':     val_mean_f1,
            }, best_ckpt_stage2)
            print(f"  ** Stage2-ready checkpoint saved: {best_ckpt_stage2}")

        # Checkpoint mỗi 4 epoch
        if (epoch + 1) % 4 == 0:
            ckpt_path = os.path.join(conf['outdir'],
                                     f'epoch{epoch+1}_model_fold{conf.fold}.pth')
            raw_state_dict = net.state_dict()
            torch.save({
                'epoch':      epoch,
                'state_dict': raw_state_dict,
                'optimizer':  optimizer.state_dict(),
            }, ckpt_path)

            ckpt_stage2_path = os.path.join(
                conf['outdir'],
                f'epoch{epoch+1}_model_stage2_ready_fold{conf.fold}.pth'
            )
            torch.save({
                'epoch':      epoch,
                'state_dict': to_stage2_compatible_state_dict(raw_state_dict),
            }, ckpt_stage2_path)

        logging.info(f"Epoch {epoch+1}: train_loss={train_loss:.5f}  "
                     f"val_loss={val_loss:.5f}  F1={100.*val_mean_f1:.2f}%")

    # ── Summary ──
    end_time = datetime.now()
    print("=" * 70)
    print(f"End time  : {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration  : {end_time - start_time}")
    print(f"Best F1   : {100.*best_f1:.2f}%  (epoch {best_epoch})")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    conf = get_config()
    set_env(conf)
    set_outdir(conf)
    set_logger(conf)
    main(conf)
