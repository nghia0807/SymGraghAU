"""
Visualize logic embedding space (CNF / SAT / UNSAT) + Multi-panel plot (5 vùng)
Dựa trực tiếp trên cơ chế Phase 2 của SymGraphAU (train_Sym_Stage_2.py).

Cách dùng (ví dụ DISFA, fold 2, vẽ toàn bộ val set):

python visualize_logic_space_phase2_eval.py ^
  --dataset DISFA ^
  --root data/DISFA ^
  --arc resnet50 ^
  --fold 2 ^
  --stage1_ckpt results/resnet50_first_stage/bs_64_seed_0_lr_0.0001/epoch20_model_fold2.pth ^
  --phase2_ckpt results/resnet50_second_stage/bs_64_seed_0_lr_0.0001/phase2_epoch20_fold2.pth ^
  --num-samples -1 ^
  --save logic_multipanel_fold2.png
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.manifold import TSNE
from torchvision import transforms

# ====== Import từ project gốc ======
from model.SymStage1 import MEFARGStage1
from dataset import DISFA, BP4D
from utils import load_state_dict
from train_Sym_Stage_2 import (
    LogicGCN,
    LogicDiscriminator,
    LogicOperatorEmbeddings,
    LogicRuleBasePySAT,
    build_logic_graph_for_sample,
    logic_triplet_loss,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 1. Transform & Dataset giống Phase 2 (val set)
# =========================================================

def image_test(crop_size=224):
    """
    Fallback transform nếu trong dataset.py không có image_test.
    Đảm bảo trả về tensor (C,H,W) để dùng .unsqueeze(0).
    """
    print("[WARN] image_test() not found in dataset.py → using fallback transform")
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_dataset(args):
    """
    Dùng đúng class dataset như Phase 2.
    Ta lấy val set (train=False, stage=1) + image_test transform.
    """
    print("==> Preparing eval dataset for visualization...")
    tfm = image_test(crop_size=args.crop_size)

    if args.dataset == "DISFA":
        ds = DISFA(
            args.root,
            train=False,
            fold=args.fold,
            transform=tfm,
            stage=1,
        )
    else:
        ds = BP4D(
            args.root,
            train=False,
            fold=args.fold,
            transform=tfm,
            stage=1,
        )

    print(f"Eval dataset: {args.dataset}, fold={args.fold}, len={len(ds)}")
    return ds


# =========================================================
# 2. Extract Logic Embeddings (CNF / SAT / UNSAT)
# =========================================================
@torch.no_grad()
def get_logic_embeddings(
    net_stage1,
    gcn,
    op_emb,
    rule_base,
    centers_au,
    centers_expr,
    emb_dim,
    dataset,
    num_samples=50,
    indices=None,
):
    """
    Với mỗi sample:
      - Stage1 → V_a, V_e, out_AU, out_Emo
      - RuleBasePySAT → (assign_s, assign_us, cnf_str)
      - build_logic_graph_for_sample:
          + CNF graph (assignment=None)
          + SAT graph (assignment = assign_s)
          + UNSAT graph (assignment = assign_us)
      - GCN → lấy embedding node global
    """

    if indices is None:
        num_samples = min(num_samples, len(dataset))
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        indices = indices.tolist()
    else:
        num_samples = len(indices)

    cnf_list, sat_list, unsat_list = [], [], []

    print(f"[INFO] Extracting logic embeddings for {num_samples} samples...")

    pbar = tqdm(indices, desc="[Embed] ", ncols=120)

    for idx in pbar:
        sample = dataset[idx]
        # dataset có thể trả về (img, y_a) hoặc (img, y_a, path)
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            img, y_a = sample[0], sample[1]
        else:
            raise RuntimeError(
                "Dataset __getitem__ phải trả về ít nhất (image, label)"
            )

        # img: (C,H,W) tensor sau transform
        if not torch.is_tensor(img):
            raise TypeError(
                f"Expected tensor image after transform, got {type(img)}"
            )

        img = img.unsqueeze(0).to(DEVICE)  # (1,C,H,W)

        if isinstance(y_a, np.ndarray):
            y_a = torch.tensor(y_a, dtype=torch.float32)
        elif not torch.is_tensor(y_a):
            raise TypeError(f"Label y_a phải là numpy hoặc tensor, nhận được {type(y_a)}")

        y_a = y_a.to(DEVICE).float()

        # ---- Stage1 forward ----
        V_a, V_e, out_AU, out_Emo = net_stage1(img)
        # global_vec = mean pooling trên V_a (giống Phase 2)
        global_vec = V_a.mean(dim=1)[0]        # (D,)
        p_expr = torch.sigmoid(out_Emo)[0]     # (N_e,)

        # ---- generate CNF + assignments (Phase 2 style) ----
        assign_s, assign_us, cnf_str = rule_base.sample_assignments_with_pysat(
            y_a, p_expr
        )

        # ---- Build graphs ----
        X_cnf, A_cnf, idx_c = build_logic_graph_for_sample(
            cnf_str, centers_au, centers_expr, global_vec,
            op_emb, emb_dim, assignment=None, device=DEVICE
        )
        X_s, A_s, idx_s = build_logic_graph_for_sample(
            cnf_str, centers_au, centers_expr, global_vec,
            op_emb, emb_dim, assignment=assign_s, device=DEVICE
        )
        X_us, A_us, idx_u = build_logic_graph_for_sample(
            cnf_str, centers_au, centers_expr, global_vec,
            op_emb, emb_dim, assignment=assign_us, device=DEVICE
        )

        # ---- Embedding using GCN (global node) ----
        z_cnf = gcn(X_cnf, A_cnf)[idx_c]   # (D,)
        z_sat = gcn(X_s, A_s)[idx_s]      # (D,)
        z_uns = gcn(X_us, A_us)[idx_u]    # (D,)

        cnf_list.append(z_cnf.cpu())
        sat_list.append(z_sat.cpu())
        unsat_list.append(z_uns.cpu())

    return (
        torch.stack(cnf_list),
        torch.stack(sat_list),
        torch.stack(unsat_list),
    )


# =========================================================
# 3. Evaluate Phase 2 trước khi vẽ
# =========================================================
from tqdm import tqdm

@torch.no_grad()
def evaluate_phase2(
    net_stage1,
    gcn,
    disc,
    op_emb,
    centers_au,
    centers_expr,
    emb_dim,
    rule_base,
    dataset,
    lambda_c=0.1,
    margin_h=1.0,
    max_samples=None,
):
    """
    Đánh giá Phase 2 giống train_Sym_Stage_2.py:
      - L_h: logic triplet loss
      - L_c: discriminator loss
      - acc_disc: độ chính xác phân biệt SAT vs UNSAT
      - triplet_satisfied_rate: tỉ lệ (d_us > d_s + margin)
    """

    net_stage1.eval()
    gcn.eval()
    disc.eval()
    op_emb.eval()

    total_Lh = 0.0
    total_Lc = 0.0
    total_acc = 0.0
    total_triplet = 0.0

    N = len(dataset)
    if max_samples is not None and max_samples > 0:
        N = min(N, max_samples)

    print(f"[EVAL] Evaluating Phase 2 on {N} samples...")

    # ==========================
    # tqdm PROGRESS BAR HERE ⭐
    # ==========================
    pbar = tqdm(range(N), desc="[EVAL Phase 2]", ncols=120)

    for idx in pbar:
        sample = dataset[idx]
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            img, y_a = sample[0], sample[1]
        else:
            raise RuntimeError("Dataset __getitem__ phải trả về (img, label, ...)")

        if not torch.is_tensor(img):
            raise TypeError(f"Expected tensor image, got {type(img)}")

        img = img.unsqueeze(0).to(DEVICE)

        if isinstance(y_a, np.ndarray):
            y_a = torch.tensor(y_a, dtype=torch.float32)
        elif not torch.is_tensor(y_a):
            raise TypeError(f"Label y_a phải là numpy hoặc tensor, nhận được {type(y_a)}")

        y_a = y_a.to(DEVICE).float()

        # ===== Stage 1 forward =====
        V_a, V_e, out_AU, out_Emo = net_stage1(img)
        global_vec = V_a.mean(dim=1)[0]
        p_expr = torch.sigmoid(out_Emo)[0]

        # ===== Generate assignments =====
        assign_s, assign_us, cnf_str = rule_base.sample_assignments_with_pysat(
            y_a, p_expr
        )

        # ===== Build graphs =====
        X_cnf, A_cnf, idx_c = build_logic_graph_for_sample(
            cnf_str, centers_au, centers_expr, global_vec, op_emb, emb_dim,
            assignment=None, device=DEVICE
        )
        X_s, A_s, idx_s = build_logic_graph_for_sample(
            cnf_str, centers_au, centers_expr, global_vec, op_emb, emb_dim,
            assignment=assign_s, device=DEVICE
        )
        X_us, A_us, idx_u = build_logic_graph_for_sample(
            cnf_str, centers_au, centers_expr, global_vec, op_emb, emb_dim,
            assignment=assign_us, device=DEVICE
        )

        # ===== Forward GCN =====
        z_cnf = gcn(X_cnf, A_cnf)[idx_c].unsqueeze(0)   # (1,D)
        z_s = gcn(X_s, A_s)[idx_s].unsqueeze(0)         # (1,D)
        z_us = gcn(X_us, A_us)[idx_u].unsqueeze(0)      # (1,D)

        # ===== L_h (triplet) =====
        L_h = logic_triplet_loss(z_cnf, z_s, z_us, margin=margin_h)

        # ===== Discriminator =====
        emb_logic = torch.cat([z_s, z_us], dim=0)   # (2,D)
        labels_logic = torch.tensor([1., 0.], device=DEVICE)
        logits_logic = disc(emb_logic)
        L_c = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_logic, labels_logic
        )

        prob = torch.sigmoid(logits_logic)
        pred = (prob > 0.5).float()
        acc_disc = (pred == labels_logic).float().mean()

        # ===== Triplet satisfied =====
        d_s = ((z_cnf - z_s) ** 2).sum(dim=1)
        d_us = ((z_cnf - z_us) ** 2).sum(dim=1)
        triplet_sat = (d_us > d_s + margin_h).float().mean()

        total_Lh += L_h.item()
        total_Lc += L_c.item()
        total_acc += acc_disc.item()
        total_triplet += triplet_sat.item()

        # ==== Update tqdm display ====
        pbar.set_postfix({
            "L_h": f"{L_h.item():.4f}",
            "L_c": f"{L_c.item():.4f}",
            "Acc": f"{acc_disc.item():.3f}",
            "Triplet": f"{triplet_sat.item():.3f}",
        })

    return {
        "L_h": total_Lh / N,
        "L_c": total_Lc / N,
        "acc_disc": total_acc / N,
        "triplet_satisfied_rate": total_triplet / N,
    }
# =========================================================
# 4. Multi-panel t-SNE plot (5 vùng)
# =========================================================
def crop_panel(ax, X2, labels, x1, x2, y1, y2, title):
    for i, lab in enumerate(labels):
        x, y = X2[i]
        if x1 <= x <= x2 and y1 <= y <= y2:
            if lab == "CNF":
                ax.scatter(x, y, c="gold", s=10, marker='x', linewidths=0.5)
            elif lab == "SAT":
                ax.scatter(x, y, c="green", s=10, marker='+', linewidths=0.5)
            else:
                # '_' là marker dạng gạch ngang nhỏ (gần giống '-')
                ax.scatter(x, y, c="red", s=10, marker='_', linewidths=0.5)
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_tsne_multi_panel(cnf_emb, sat_emb, unsat_emb, X2, save_path="logic_multipanel.png"):
    """
    - Panel 1: full t-SNE space
    - Panel 2: CNF region (bao phủ CNF, thường gần SAT)
    - Panel 3: UNSAT region
    - Panel 4: Border region (vùng giữa SAT & UNSAT)
    - Panel 5: Abnormal region (UNSAT gần cluster SAT nhất)
    """
    N_cnf = len(cnf_emb)
    N_sat = len(sat_emb)
    N_uns = len(unsat_emb)

    labels = ["CNF"] * N_cnf + ["SAT"] * N_sat + ["UNSAT"] * N_uns

    fig, axes = plt.subplots(1, 5, figsize=(28, 6))

    # ------------------ PANEL 1: FULL SPACE ------------------
    ax = axes[0]
    for i, lab in enumerate(labels):
        if lab == "CNF":
            ax.scatter(
                X2[i, 0], X2[i, 1],
                c="gold", s=10, marker='x', linewidths=0.5,
                label="CNF" if i == 0 else ""
            )
        elif lab == "SAT":
            ax.scatter(
                X2[i, 0], X2[i, 1],
                c="green", s=10, marker='+', linewidths=0.5,
                label="SAT" if i == N_cnf else ""
            )
        else:
            ax.scatter(
                X2[i, 0], X2[i, 1],
                c="red", s=10, marker='_', linewidths=0.5,
                label="UNSAT" if i == (N_cnf + N_sat) else ""
            )
    ax.set_title("Full t-SNE Space")
    ax.legend()

    # ------------------ PANEL 2: CNF–SAT REGION ------------------
    cnf_xy = X2[:N_cnf]
    cx1, cy1 = cnf_xy.min(axis=0) - 0.05
    cx2, cy2 = cnf_xy.max(axis=0) + 0.05
    crop_panel(axes[1], X2, labels, cx1, cx2, cy1, cy2, "CNF–SAT Region")

    # ------------------ PANEL 3: UNSAT REGION ------------------
    uns_xy = X2[N_cnf + N_sat:]
    ux1, uy1 = uns_xy.min(axis=0) - 0.05
    ux2, uy2 = uns_xy.max(axis=0) + 0.05
    crop_panel(axes[2], X2, labels, ux1, ux2, uy1, uy2, "UNSAT Region")

    # ------------------ PANEL 4: BORDER REGION ------------------
    sat_xy = X2[N_cnf:N_cnf + N_sat]
    uns_xy = X2[N_cnf + N_sat:]
    mid = (sat_xy.mean(axis=0) + uns_xy.mean(axis=0)) / 2
    r = np.linalg.norm(sat_xy.mean(axis=0) - uns_xy.mean(axis=0)) * 0.6
    crop_panel(
        axes[3], X2, labels,
        mid[0] - r, mid[0] + r,
        mid[1] - r, mid[1] + r,
        "Border Region"
    )

    # ------------------ PANEL 5: ABNORMAL REGION ------------------
    # auto detect UNSAT gần cluster SAT nhất
    sat_c = sat_xy.mean(axis=0)
    dist_uns = np.linalg.norm(uns_xy - sat_c, axis=1)
    idx_min = np.argmin(dist_uns)
    abnormal_center = uns_xy[idx_min]

    r2 = 0.1
    ax5 = axes[4]
    crop_panel(
        ax5, X2, labels,
        abnormal_center[0] - r2,
        abnormal_center[0] + r2,
        abnormal_center[1] - r2,
        abnormal_center[1] + r2,
        "Abnormal Region"
    )

    # Vẽ khung vùng "abnormal" trên Panel 1
    rect = patches.Rectangle(
        (abnormal_center[0] - r2, abnormal_center[1] - r2),
        2 * r2, 2 * r2,
        linewidth=2, edgecolor='blue', linestyle='--', fill=False
    )
    axes[0].add_patch(rect)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[SAVED] Multi-panel visualization → {save_path}")
    plt.show()


# =========================================================
# 5. MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visualize logic embedding space (CNF / SAT / UNSAT) dựa trên Phase 2"
    )

    parser.add_argument("--dataset", type=str, default="DISFA", choices=["DISFA", "BP4D"])
    parser.add_argument(
        "--root", type=str, default="data/DISFA",
        help="Dataset root (giống conf.dataset_path). Ví dụ: data/DISFA"
    )
    parser.add_argument("--arc", type=str, default="resnet50")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=224)

    parser.add_argument("--val2", type=int, default=0)

    parser.add_argument(
        "--stage1_ckpt", type=str, required=True,
        help="Checkpoint Phase 1 (Stage1), ví dụ: epoch20_model_fold2.pth"
    )
    parser.add_argument(
        "--phase2_ckpt", type=str, required=True,
        help="Checkpoint Phase 2, ví dụ: phase2_epoch20_fold2.pth"
    )
    parser.add_argument(
        "--num-samples", type=int, default=-1,
        help="Số sample để visualize; <=0 = dùng toàn bộ dataset"
    )
    parser.add_argument(
        "--save", type=str, default="logic_multipanel.png",
        help="Đường dẫn file PNG để lưu plot"
    )

    args = parser.parse_args()

    # ------------------ Load Stage1 ------------------
    print("[LOAD] Stage1 checkpoint:", args.stage1_ckpt)
    net_stage1 = MEFARGStage1(num_aus=8, backbone=args.arc, num_expr=7)
    net_stage1 = load_state_dict(net_stage1, args.stage1_ckpt)
    net_stage1 = net_stage1.to(DEVICE).eval()

    # ------------------ Load Phase2 ------------------
    print("[LOAD] Phase2 checkpoint:", args.phase2_ckpt)
    ckpt_p2 = torch.load(args.phase2_ckpt, map_location="cpu")

    centers_au = ckpt_p2["centers_au"].to(DEVICE)
    centers_expr = ckpt_p2["centers_expr"].to(DEVICE)
    emb_dim = centers_au.size(1)

    gcn = LogicGCN(in_dim=emb_dim + 4, hidden_dim=emb_dim, out_dim=emb_dim)
    gcn.load_state_dict(ckpt_p2["gcn_state_dict"])
    gcn = gcn.to(DEVICE).eval()

    disc = LogicDiscriminator(emb_dim=emb_dim)
    disc.load_state_dict(ckpt_p2["disc_state_dict"])
    disc = disc.to(DEVICE).eval()

    op_emb = LogicOperatorEmbeddings(emb_dim)
    op_emb.load_state_dict(ckpt_p2["op_emb_state_dict"])
    op_emb = op_emb.to(DEVICE).eval()

    # RuleBase mới (tham số giống Phase 2)
    rule_base = LogicRuleBasePySAT(num_aus=8, num_expr=7)

    # ------------------ Dataset (val set) ------------------
    dataset = get_eval_dataset(args)

    # ------------------ Phase 2 evaluation ------------------
    if args.val2 == 1:
        print("\n[PHASE 2 EVALUATION]")
        metrics = evaluate_phase2(
            net_stage1,
            gcn,
            disc,
            op_emb,
            centers_au,
            centers_expr,
            emb_dim,
            rule_base,
            dataset,
            lambda_c=0.1,
            margin_h=1.0,
            max_samples=None,   # có thể set nhỏ hơn để chạy nhanh
        )
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    # ------------------ Chọn sample để vẽ ------------------
    if args.num_samples <= 0 or args.num_samples > len(dataset):
        indices = list(range(len(dataset)))   # toàn bộ dataset
    else:
        indices = random.sample(range(len(dataset)), args.num_samples)

    # ------------------ Extract embeddings ------------------
    cnf_emb, sat_emb, unsat_emb = get_logic_embeddings(
        net_stage1, gcn, op_emb, rule_base,
        centers_au, centers_expr, emb_dim,
        dataset,
        num_samples=len(indices),
        indices=indices,
    )

    # ------------------ T-SNE ------------------
    print("[TSNE] Reducing dimension...")
    X = torch.cat([cnf_emb, sat_emb, unsat_emb], dim=0).numpy()
    tsne = TSNE(
        n_components=2,
        learning_rate="auto",
        perplexity=min(30, max(5, X.shape[0] // 3)),
        init="random"
    )
    X2 = tsne.fit_transform(X)

    # Chuẩn hóa về [0, 1] cho đẹp (giống paper)
    X2_min = X2.min(axis=0, keepdims=True)
    X2_max = X2.max(axis=0, keepdims=True)
    X2 = (X2 - X2_min) / (X2_max - X2_min + 1e-9)

    # ------------------ Multi-panel plot ------------------
    plot_tsne_multi_panel(cnf_emb, sat_emb, unsat_emb, X2, save_path=args.save)


if __name__ == "__main__":
    main()
