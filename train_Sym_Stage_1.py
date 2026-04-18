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

from model.SymStage1 import MEFARGStage1
from dataset import *
from utils import *
from conf import get_config,set_logger,set_outdir,set_env


def get_dataloader(conf):
    print('==> Preparing data...')
    if conf.dataset == 'BP4D':
        trainset = BP4D(conf.dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 1)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        valset = BP4D(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 1)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    elif conf.dataset == 'DISFA':
        trainset = DISFA(conf.dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 1)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        valset = DISFA(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 1)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    return train_loader, val_loader, len(trainset), len(valset)

def au_to_expr_pseudo(Y_a: torch.Tensor,
                      M_AE: torch.Tensor,
                      neutral_index: int) -> torch.Tensor:
    """
    Y_a: (B, N_a)  - nhãn AU (0/1) hoặc xác suất sau sigmoid
    M_AE: (N_a, N_e) - ma trận AU-Expression
    neutral_index: int - chỉ số class 'Neutral' trong trục expression

    return: Y_e (B, N_e), one-hot pseudo-label expression
    """
    B, N_a = Y_a.shape
    N_e = M_AE.shape[1]

    # Đảm bảo dùng float cho nhân ma trận
    Y_a_float = Y_a.float()

    # (1) Tính score theo Eq.(1): s = Y^a * M_{A-E}
    scores = Y_a_float @ M_AE    # (B, N_e)

    # (2) Chọn ke = argmax(score) cho mỗi sample
    ke = scores.argmax(dim=1)    # (B,)

    # (3) Xử lý các mẫu không có AU nào kích hoạt → Neutral
    neutral_mask = (Y_a_float.sum(dim=1) == 0)   # (B,)
    ke[neutral_mask] = neutral_index

    # (4) Tạo one-hot Y^e theo Eq.(2)
    Y_e = torch.zeros(B, N_e, device=Y_a.device, dtype=Y_a_float.dtype)
    Y_e.scatter_(1, ke.unsqueeze(1), 1.0)

    return Y_e  # shape (B, N_e)

M_AE_np = np.load(r"matrixMAE\M_AE_DISFA.npy")
M_AE = torch.from_numpy(M_AE_np).float()

# Train
def train(conf, net, train_loader, optimizer, epoch, criterion, criterion_Em):
    losses = AverageMeter()
    net.train()
    train_loader_len = len(train_loader)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (inputs, targets) in enumerate(pbar):

        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate,
                             batch_idx, train_loader_len)

        targets = targets.float()

        if torch.cuda.is_available():
            inputs  = inputs.cuda()
            targets = targets.cuda()

        # ----- Pseudo Label -----
        targets_Emo = au_to_expr_pseudo(targets, M_AE, neutral_index=6)
        if torch.cuda.is_available():
            targets_Emo = targets_Emo.cuda()

        optimizer.zero_grad()

        # ----- Forward -----
        V_a, V_e, outputs_AU, outputs_Emo = net(inputs)

        # ----- Loss -----
        L_wa = criterion(outputs_AU, targets)              # AU loss (3)
        L_we = criterion_Em(outputs_Emo, targets_Emo)      # Emotion loss (4)

        gamma = 1.0
        loss = L_wa + gamma * L_we

        # ----- Backprop -----
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))

        # ====== TQDM cập nhật mô tả ======
        pbar.set_postfix({
            'L_wa': f"{L_wa.item():.4f}",
            'L_we': f"{L_we.item():.4f}",
            'L_jf': f"{loss.item():.4f}",
        })

    return losses.avg



# Val
def val(net, val_loader, criterion):
    losses = AverageMeter()
    net.eval()
    statistics_list = None

    pbar = tqdm(val_loader, desc="Val")

    for batch_idx, (inputs, targets) in enumerate(pbar):
        with torch.no_grad():
            targets = targets.float()
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            # Model mới: trả về 4 giá trị
            # V_a: (B, N_a, EMB_DIM)
            # V_e: (B, N_e, EMB_DIM)
            # p_a: (B, N_a)  -> dùng cho AU eval
            # p_e: (B, N_e)
            V_a, V_e, outputs_AU, outputs_Emo = net(inputs)

            # AU loss
            loss = criterion(outputs_AU, targets)
            losses.update(loss.item(), inputs.size(0))

            # Thống kê AU performance (dùng outputs_AU)
            update_list = statistics(outputs_AU, targets.detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)

            # Cập nhật tqdm
            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)

    return losses.avg, mean_f1_score, f1_score_list, mean_acc, acc_list

from datetime import datetime

def main(conf):


    if conf.dataset == 'BP4D':
        dataset_info = BP4D_infolist
    elif conf.dataset == 'DISFA':
        dataset_info = DISFA_infolist

    start_epoch = 0
    # data
    train_loader,val_loader,train_data_num,val_data_num = get_dataloader(conf)
    train_weight = torch.from_numpy(np.loadtxt(os.path.join(conf.dataset_path, 'list', conf.dataset+'_weight_fold'+str(conf.fold)+'.txt')))

    weight_path = os.path.join(conf.dataset_path, 'list',
                           f'{conf.dataset}_train_weight_fold{conf.fold}.txt')  # chú ý: thêm 'train_'
    train_weight = np.loadtxt(weight_path)
    train_weight = torch.from_numpy(train_weight).float()
    # 🔍 Kiểm tra trọng số WAL
    print(f"[WAL] weight_path = {weight_path}")
    print(f"[WAL] w = {train_weight.tolist()}, sum={float(train_weight.sum()):.6f}")

    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold, conf.N_fold, val_data_num))

    net = MEFARGStage1(num_aus=conf.num_classes, backbone=conf.arc, num_expr=7)
    # resume
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
        train_weight = train_weight.cuda()
        global M_AE
        M_AE = M_AE.cuda()

    criterion = WeightedAsymmetricLoss(weight=train_weight)
    criterion_Em = ExpressionBCELoss()
    optimizer = optim.AdamW(net.parameters(),  betas=(0.9, 0.999), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    print('the init learning rate is ', conf.learning_rate)

    start_time = datetime.now()
    print("Start time:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    #train and val
    for epoch in range(start_epoch, conf.epochs):

        print("--Weight will be Saved At", os.path.join(conf['outdir'], 'epoch' + str(epoch + 1) + '_model_fold' + str(conf.fold) + '.pth'))

        logging.info(f"[CFG] arc={conf.arc} K={conf.neighbor_num} metric={conf.metric} "
                 f"bs={conf.batch_size} lr0={conf.learning_rate} wd={conf.weight_decay}")

        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        train_loss = train(conf,net,train_loader,optimizer,epoch,criterion, criterion_Em)

        val_loss, val_mean_f1_score, val_f1_score, val_mean_acc, val_acc = val(net, val_loader, criterion)

       

        # log
        infostr = {'Epoch:  {}   train_loss: {:.5f}  val_loss: {:.5f}  val_mean_f1_score {:.2f},val_mean_acc {:.2f}'
                .format(epoch + 1, train_loss, val_loss, 100.* val_mean_f1_score, 100.* val_mean_acc)}
        
        print(infostr)

        # logging.info(infostr)
        # infostr = {'F1-score-list:'}
        # logging.info(infostr)
        # infostr = dataset_info(val_f1_score)
        # logging.info(infostr)
        # infostr = {'Acc-list:'}
        # logging.info(infostr)
        # infostr = dataset_info(val_acc)
        # logging.info(infostr)

        # save checkpoints
        if (epoch+1) % 4 == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            
            torch.save(checkpoint, os.path.join(conf['outdir'], 'epoch' + str(epoch + 1) + '_model_fold' + str(conf.fold) + '.pth'))

        # checkpoint = {
        #     'epoch': epoch,
        #     'state_dict': net.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }
        # torch.save(checkpoint, os.path.join(conf['outdir'], 'cur_model_fold' + str(conf.fold) + '.pth'))


        
    end_time = datetime.now()
    print("End time:", end_time.strftime("%Y-%m-%d %H:%M:%S"))

    # Calculate duration
    duration = end_time - start_time
    print("Duration:", duration)

# ---------------------------------------------------------------------------------


if __name__=="__main__":
    conf = get_config()
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)
