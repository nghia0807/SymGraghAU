import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging

from model.SymStage1 import MEFARGStage1
from dataset import *
from utils import *
from conf import get_config, set_logger, set_outdir, set_env

# Re-use logic modules from Phase 2
from train_Sym_Stage_2 import (
    LogicGCN,
    LogicOperatorEmbeddings,
    LogicRuleBasePySAT,
    AU_NAME_TO_IDX,
    EXPR_NAME_TO_IDX,
)


# ============================================================
# 0. DataLoader (giống Phase 1 & 2)
# ============================================================

def get_dataloader(conf):
    print('==> Preparing data (Phase 3)...')
    loader_kwargs = {
        "batch_size": conf.batch_size,
        "num_workers": conf.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if conf.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    if conf.dataset == 'BP4D':
        trainset = BP4D(
            conf.dataset_path,
            train=True,
            fold=conf.fold,
            transform=image_train(crop_size=conf.crop_size),
            crop_size=conf.crop_size,
            stage=1,
        )
        train_loader = DataLoader(
            trainset,
            shuffle=True,
            **loader_kwargs,
        )
        valset = BP4D(
            conf.dataset_path,
            train=False,
            fold=conf.fold,
            transform=image_test(crop_size=conf.crop_size),
            stage=1,
        )
        val_loader = DataLoader(
            valset,
            shuffle=False,
            **loader_kwargs,
        )

    elif conf.dataset == 'DISFA':
        trainset = DISFA(
            conf.dataset_path,
            train=True,
            fold=conf.fold,
            transform=image_train(crop_size=conf.crop_size),
            crop_size=conf.crop_size,
            stage=1,
        )
        train_loader = DataLoader(
            trainset,
            shuffle=True,
            **loader_kwargs,
        )
        valset = DISFA(
            conf.dataset_path,
            train=False,
            fold=conf.fold,
            transform=image_test(crop_size=conf.crop_size),
            stage=1,
        )
        val_loader = DataLoader(
            valset,
            shuffle=False,
            **loader_kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset: {conf.dataset}")

    return train_loader, val_loader, len(trainset), len(valset)


# ============================================================
# 1. Phase 3 loss: L_total = L_wa + mu * L_l
#    L_l: MSE giữa global node của G_cnf(I_t) và G_p(I_t)
# ============================================================

_CNF_TOPOLOGY_CACHE = {}


def _cnf_to_hashable_key(cnf_clauses_str):
    return tuple(tuple(clause) for clause in cnf_clauses_str)


def _parse_literal(lit_str):
    lit_str = lit_str.strip()
    neg = lit_str.startswith("¬")
    if neg:
        name = lit_str[1:]
    else:
        name = lit_str
    return name, neg


def _get_or_build_graph_topology(cnf_clauses_str, num_aus, num_expr, device):
    """Build and cache static graph topology for a CNF clause pattern."""
    key = (num_aus, num_expr, _cnf_to_hashable_key(cnf_clauses_str), str(device))
    cached = _CNF_TOPOLOGY_CACHE.get(key)
    if cached is not None:
        return cached

    num_clause = len(cnf_clauses_str)
    idx_and = num_aus + num_expr + num_clause
    idx_global = idx_and + 1
    num_nodes = idx_global + 1

    A = torch.zeros(num_nodes, num_nodes, device=device)

    # OR nodes connect to AND and literals.
    for c_idx, clause in enumerate(cnf_clauses_str):
        or_idx = num_aus + num_expr + c_idx
        A[or_idx, idx_and] = 1.0
        A[idx_and, or_idx] = 1.0

        for lit in clause:
            name, _ = _parse_literal(lit)
            if name in AU_NAME_TO_IDX:
                lit_idx = AU_NAME_TO_IDX[name]
                if lit_idx >= num_aus:
                    continue
            elif name in EXPR_NAME_TO_IDX:
                expr_j = EXPR_NAME_TO_IDX[name]
                if expr_j >= num_expr:
                    continue
                lit_idx = num_aus + expr_j
            else:
                continue

            A[or_idx, lit_idx] = 1.0
            A[lit_idx, or_idx] = 1.0

    # Global node connects to all nodes.
    A[idx_global, :] = 1.0
    A[:, idx_global] = 1.0
    A[idx_global, idx_global] = 0.0

    # Static one-hot type code for all nodes.
    # [global, and, or, leaf]
    type_code = torch.zeros(num_nodes, 4, device=device)
    type_code[: num_aus + num_expr, 3] = 1.0
    if num_clause > 0:
        type_code[num_aus + num_expr: idx_and, 2] = 1.0
    type_code[idx_and, 1] = 1.0
    type_code[idx_global, 0] = 1.0

    cached = (A, type_code, idx_global, num_clause)
    _CNF_TOPOLOGY_CACHE[key] = cached
    return cached


def _build_graph_features_pair(
    type_code,
    idx_global,
    num_clause,
    centers_au,
    centers_expr,
    p_a,
    p_expr,
    global_vec,
    op_emb_module,
    emb_dim,
):
    """Build feature matrices for G_cnf and G_p with shared topology."""
    device = centers_au.device
    num_aus = centers_au.size(0)
    num_expr = centers_expr.size(0)
    num_nodes = type_code.size(0)

    x_cnf = torch.zeros(num_nodes, emb_dim + 4, device=device)
    x_pred = torch.zeros_like(x_cnf)

    x_cnf[:, :4] = type_code
    x_pred[:, :4] = type_code

    # Leaf AU nodes.
    x_cnf[:num_aus, 4:] = centers_au
    x_pred[:num_aus, 4:] = centers_au * p_a.unsqueeze(1)

    # Leaf expression nodes.
    expr_start = num_aus
    expr_end = num_aus + num_expr
    x_cnf[expr_start:expr_end, 4:] = centers_expr
    x_pred[expr_start:expr_end, 4:] = centers_expr * p_expr.unsqueeze(1)

    # OR nodes.
    or_start = expr_end
    or_end = or_start + num_clause
    if num_clause > 0:
        or_emb = op_emb_module.or_emb.unsqueeze(0)
        x_cnf[or_start:or_end, 4:] = or_emb.expand(num_clause, -1)
        x_pred[or_start:or_end, 4:] = or_emb.expand(num_clause, -1)

    # AND node.
    idx_and = or_end
    x_cnf[idx_and, 4:] = op_emb_module.and_emb
    x_pred[idx_and, 4:] = op_emb_module.and_emb

    # Global node.
    x_cnf[idx_global, 4:] = global_vec
    x_pred[idx_global, 4:] = global_vec

    return x_cnf, x_pred

def compute_logic_loss_for_batch(
    V_a,
    outputs_AU,
    outputs_Emo,
    targets,
    gcn,
    op_emb_module,
    centers_au,
    centers_expr,
    emb_dim,
    rule_base_pysat,
    mu_l=0.1,
):
    """
    V_a: (B, N_a, D) – AU embeddings từ Stage 1
    outputs_AU: (B, N_a) – logits AU
    outputs_Emo: (B, N_e) – logits expression
    targets: (B, N_a) – GT AU (0/1)
    """
    device = V_a.device
    B, N_a, D = V_a.shape

    global_vec_batch = V_a.mean(dim=1)                 # (B, D)
    p_a_batch = torch.sigmoid(outputs_AU)              # (B, N_a)
    p_expr_batch = torch.sigmoid(outputs_Emo)          # (B, N_e)

    p_expr_batch_cpu = p_expr_batch.detach().to("cpu")
    targets_cpu = targets.detach().to("cpu")

    a_blocks = []
    x_cnf_blocks = []
    x_pred_blocks = []
    global_indices = []
    offset = 0

    for b in range(B):
        y_a_b = targets_cpu[b]                         # (N_a,), CPU to avoid GPU sync in PySAT rule build
        global_vec_b = global_vec_batch[b]             # (D,)
        p_a_b = p_a_batch[b]                           # (N_a,)
        p_expr_b = p_expr_batch[b]                     # (N_e,)
        p_expr_b_cpu = p_expr_batch_cpu[b]

        # ----- Sinh CNF paradigm cho sample (Phase 2 style) -----
        # dùng GT AU + p_expr_b (expression prob) → CNF_t(I_t)
        _, _, cnf_str_b = rule_base_pysat.sample_assignments_with_pysat(
            y_a_b, p_expr_b_cpu
        )

        # Build topology once and reuse for both G_cnf and G_p.
        A_b, type_code_b, idx_g_b, num_clause_b = _get_or_build_graph_topology(
            cnf_str_b,
            device=device,
            num_aus=centers_au.size(0),
            num_expr=centers_expr.size(0),
        )

        X_cnf_b, X_pred_b = _build_graph_features_pair(
            type_code_b,
            idx_g_b,
            num_clause_b,
            centers_au,
            centers_expr,
            p_a_b,
            p_expr_b,
            global_vec_b,
            op_emb_module,
            emb_dim,
        )

        a_blocks.append(A_b)
        x_cnf_blocks.append(X_cnf_b)
        x_pred_blocks.append(X_pred_b)
        global_indices.append(offset + idx_g_b)
        offset += A_b.size(0)

    if len(a_blocks) == 1:
        A_big = a_blocks[0]
    else:
        A_big = torch.block_diag(*a_blocks)

    X_cnf_big = torch.cat(x_cnf_blocks, dim=0)
    X_pred_big = torch.cat(x_pred_blocks, dim=0)

    # Two batched GCN passes instead of 2*B per-sample passes.
    Q_cnf_big = gcn(X_cnf_big, A_big)
    Q_pred_big = gcn(X_pred_big, A_big)

    global_idx = torch.tensor(global_indices, device=device, dtype=torch.long)
    z_cnf = Q_cnf_big.index_select(0, global_idx)
    z_pred = Q_pred_big.index_select(0, global_idx)

    L_l = F.mse_loss(z_cnf, z_pred)
    return mu_l * L_l, L_l.detach().item()


def train_phase3(
    conf,
    net,
    train_loader,
    optimizer,
    epoch,
    criterion_wa,
    gcn,
    op_emb_module,
    centers_au,
    centers_expr,
    emb_dim,
    rule_base_pysat,
    mu_l=0.1,
):
    net.train()
    gcn.eval()
    op_emb_module.eval()

    device = next(net.parameters()).device
    total_meter = AverageMeter()
    wa_meter = AverageMeter()
    logic_meter = AverageMeter()

    train_loader_len = len(train_loader)
    pbar = tqdm(train_loader, desc=f"[Phase3 Train] Epoch {epoch}")
    use_cuda = torch.cuda.is_available()

    for batch_idx, (inputs, targets) in enumerate(pbar):
        adjust_learning_rate(
            optimizer,
            epoch,
            conf.epochs,
            conf.learning_rate,
            batch_idx,
            train_loader_len,
        )

        targets = targets.float()
        if use_cuda:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ----- Forward Stage 1 (đang fine-tune) -----
        V_a, V_e, outputs_AU, outputs_Emo = net(inputs)

        # ----- AU detection loss (Weighted Asymmetric Loss) -----
        L_wa = criterion_wa(outputs_AU, targets)

        # ----- Logic loss L_l (knowledge regularizer) -----
        muL_l, raw_L_l = compute_logic_loss_for_batch(
            V_a,
            outputs_AU,
            outputs_Emo,
            targets,
            gcn,
            op_emb_module,
            centers_au,
            centers_expr,
            emb_dim,
            rule_base_pysat,
            mu_l=mu_l,
        )

        loss = L_wa + muL_l
        loss.backward()
        optimizer.step()

        total_meter.update(loss.item(), inputs.size(0))
        wa_meter.update(L_wa.item(), inputs.size(0))
        logic_meter.update(raw_L_l, inputs.size(0))

        pbar.set_postfix({
            "L_total": f"{total_meter.avg:.4f}",
            "L_wa": f"{wa_meter.avg:.4f}",
            "L_l": f"{logic_meter.avg:.4f}",
        })

    return total_meter.avg, wa_meter.avg, logic_meter.avg


@torch.no_grad()
def val_phase3(
    conf,
    net,
    val_loader,
    criterion_wa,
    gcn,
    op_emb_module,
    centers_au,
    centers_expr,
    emb_dim,
    rule_base_pysat,
    mu_l=0.1,
):
    net.eval()
    gcn.eval()
    op_emb_module.eval()

    device = next(net.parameters()).device
    total_meter = AverageMeter()
    wa_meter = AverageMeter()
    logic_meter = AverageMeter()

    statistics_list = None

    pbar = tqdm(val_loader, desc="[Phase3 Val]")
    use_cuda = torch.cuda.is_available()

    for batch_idx, (inputs, targets) in enumerate(pbar):
        targets = targets.float()
        if use_cuda:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

        V_a, V_e, outputs_AU, outputs_Emo = net(inputs)

        # AU detection loss
        L_wa = criterion_wa(outputs_AU, targets)

        # Logic loss
        muL_l, raw_L_l = compute_logic_loss_for_batch(
            V_a,
            outputs_AU,
            outputs_Emo,
            targets,
            gcn,
            op_emb_module,
            centers_au,
            centers_expr,
            emb_dim,
            rule_base_pysat,
            mu_l=mu_l,
        )

        loss = L_wa + muL_l

        total_meter.update(loss.item(), inputs.size(0))
        wa_meter.update(L_wa.item(), inputs.size(0))
        logic_meter.update(raw_L_l, inputs.size(0))

        # AU metrics: F1, Acc (giống Phase 1)
        update_list = statistics(outputs_AU, targets.detach(), 0.5)
        statistics_list = update_statistics_list(statistics_list, update_list)

        pbar.set_postfix({
            "L_total": f"{total_meter.avg:.4f}",
            "L_wa": f"{wa_meter.avg:.4f}",
            "L_l": f"{logic_meter.avg:.4f}",
        })

    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)

    return (
        total_meter.avg,
        wa_meter.avg,
        logic_meter.avg,
        mean_f1_score,
        f1_score_list,
        mean_acc,
        acc_list,
    )


# ============================================================
# 2. main(): load Phase 1 & Phase 2, rồi fine-tune (Phase 3)
# ============================================================

def main(conf):

    if conf.dataset == "BP4D":
        dataset_info = BP4D_infolist
    elif conf.dataset == "DISFA":
        dataset_info = DISFA_infolist
    else:
        raise ValueError(f"Unknown dataset: {conf.dataset}")

    # ---------- Dataloader ----------
    train_loader, val_loader, train_data_num, val_data_num = get_dataloader(conf)
    logging.info(
        "Fold: [{} | {}  val_data_num: {} ]".format(
            conf.fold, conf.N_fold, val_data_num
        )
    )

    # ---------- WAL weight ----------
    weight_path = os.path.join(
        conf.dataset_path, "list", f"{conf.dataset}_train_weight_fold{conf.fold}.txt"
    )
    train_weight = np.loadtxt(weight_path)
    train_weight = torch.from_numpy(train_weight).float()
    print(f"[WAL] weight_path = {weight_path}")
    print(f"[WAL] w = {train_weight.tolist()}, sum={float(train_weight.sum()):.6f}")

    # ---------- Stage 1 (AU detector) ----------
    net = MEFARGStage1(num_aus=conf.num_classes, backbone=conf.arc, num_expr=7)

    # conf.resume: checkpoint từ Phase 1 (JFL)
    if getattr(conf, "resume", "") != "":
        logging.info(f"[Phase3] Resume Stage1 from | {conf.resume} ]")
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
        train_weight = train_weight.cuda()

    criterion_wa = WeightedAsymmetricLoss(weight=train_weight)

    optimizer = optim.AdamW(
        net.parameters(),
        betas=(0.9, 0.999),
        lr=conf.learning_rate,
        weight_decay=conf.weight_decay,
    )
    print("Phase 3 init learning rate:", conf.learning_rate)

    # ---------- Stage 2 (logic embedder) ----------
    # conf.resume_phase2: checkpoint Phase 2 đã train xong
    resume_phase2 = getattr(conf, "resume_phase2", "")
    assert resume_phase2 != "", (
        "For Phase 3, please provide Phase 2 checkpoint via conf.resume_phase2 "
        "(e.g. --resume-phase2 path/to/phase2_epochX_foldY.pth)"
    )
    logging.info(f"[Phase3] Load Phase2 (logic embedder) from | {resume_phase2} ]")

    ckpt_logic = torch.load(resume_phase2, map_location="cpu")
    centers_au = ckpt_logic["centers_au"]
    centers_expr = ckpt_logic["centers_expr"]
    emb_dim = centers_au.shape[1]

    in_dim = emb_dim + 4
    hidden_dim = emb_dim
    out_dim = emb_dim

    gcn = LogicGCN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    op_emb_module = LogicOperatorEmbeddings(emb_dim=emb_dim)

    gcn.load_state_dict(ckpt_logic["gcn_state_dict"])
    op_emb_module.load_state_dict(ckpt_logic["op_emb_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gcn = gcn.to(device)
    op_emb_module = op_emb_module.to(device)
    centers_au = centers_au.to(device)
    centers_expr = centers_expr.to(device)

    # Freeze logic embedder params trong Phase 3
    for p in gcn.parameters():
        p.requires_grad = False
    for p in op_emb_module.parameters():
        p.requires_grad = False

    # ---------- Rule base (PySAT) ----------
    rule_base_pysat = LogicRuleBasePySAT(num_aus=conf.num_classes, num_expr=7)

    # Hệ số cho logic loss L_l (mu trong Eq.(12))
    mu_l = getattr(conf, "lambda_logic", 0.1)

    # Validation và checkpoint interval:
    # default giữ nguyên hành vi cũ (val mỗi epoch, save mỗi 2 epoch).
    val_interval = max(1, int(os.getenv("PHASE3_VAL_INTERVAL", "1")))
    save_interval = max(1, int(os.getenv("PHASE3_SAVE_INTERVAL", "2")))

    start_time = datetime.now()
    print("Start time:", start_time.strftime("%Y-%m-%d %H:%M:%S"))

    for epoch in range(conf.epochs):
        save_path = os.path.join(
            conf["outdir"],
            f"phase3_epoch{epoch+1}_fold{conf.fold}.pth",
        )
        print("--Phase3 Weights will be saved at", save_path)

        logging.info(
            f"[Phase3 CFG] arc={conf.arc} K={conf.neighbor_num} "
            f"metric={conf.metric} bs={conf.batch_size} "
            f"lr0={conf.learning_rate} wd={conf.weight_decay} "
            f"mu_l={mu_l}"
        )

        lr = optimizer.param_groups[0]["lr"]
        logging.info(
            "Phase3 Epoch: [{} | {} LR: {} ]".format(
                epoch + 1, conf.epochs, lr
            )
        )

        train_L_total, train_L_wa, train_L_l = train_phase3(
            conf,
            net,
            train_loader,
            optimizer,
            epoch,
            criterion_wa,
            gcn,
            op_emb_module,
            centers_au,
            centers_expr,
            emb_dim,
            rule_base_pysat,
            mu_l=mu_l,
        )

        do_val = ((epoch + 1) % val_interval == 0) or ((epoch + 1) == conf.epochs)
        if do_val:
            (
                val_L_total,
                val_L_wa,
                val_L_l,
                val_mean_f1,
                val_f1_list,
                val_mean_acc,
                val_acc_list,
            ) = val_phase3(
                conf,
                net,
                val_loader,
                criterion_wa,
                gcn,
                op_emb_module,
                centers_au,
                centers_expr,
                emb_dim,
                rule_base_pysat,
                mu_l=mu_l,
            )

            infostr = {
                "Phase3 Epoch: {}  train_L_total: {:.5f}  val_L_total: {:.5f}  "
                "train_L_wa: {:.5f}  train_L_l: {:.5f}  val_L_wa: {:.5f}  val_L_l: {:.5f}  "
                "val_mean_f1: {:.2f}  val_mean_acc: {:.2f}".format(
                    epoch + 1,
                    train_L_total,
                    val_L_total,
                    train_L_wa,
                    train_L_l,
                    val_L_wa,
                    val_L_l,
                    100.0 * val_mean_f1,
                    100.0 * val_mean_acc,
                )
            }
        else:
            infostr = {
                "Phase3 Epoch: {}  train_L_total: {:.5f}  train_L_wa: {:.5f}  "
                "train_L_l: {:.5f}  val: skipped (interval={})".format(
                    epoch + 1,
                    train_L_total,
                    train_L_wa,
                    train_L_l,
                    val_interval,
                )
            }

        print(infostr)
        logging.info(infostr)

        # (Tuỳ chọn) log chi tiết từng AU
        # logging.info("F1-score-list:")
        # logging.info(dataset_info(val_f1_list))
        # logging.info("Acc-list:")
        # logging.info(dataset_info(val_acc_list))

        # ----- Save checkpoint -----
        if (epoch + 1) % save_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "gcn_state_dict": gcn.state_dict(),
                "op_emb_state_dict": op_emb_module.state_dict(),
                "centers_au": centers_au,
                "centers_expr": centers_expr,
            }
            torch.save(checkpoint, save_path)

    end_time = datetime.now()
    print("End time:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Duration:", end_time - start_time)


# ============================================================
# 3. Run
# ============================================================

if __name__ == "__main__":
    conf = get_config()
    set_env(conf)
    set_outdir(conf)
    set_logger(conf)
    main(conf)
