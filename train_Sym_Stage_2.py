import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging

# PySAT
from pysat.formula import CNF
from pysat.solvers import Minisat22

from model.SymStage1 import MEFARGStage1
from dataset import *
from utils import *
from conf import get_config, set_logger, set_outdir, set_env

import csv


# ============================================================
# 0. DataLoader tái dùng như Phase 1
# ============================================================

def get_dataloader(conf):
    print('==> Preparing data (Phase 2)...')
    if conf.dataset == 'BP4D':
        trainset = BP4D(
            conf.dataset_path,
            train=True,
            fold=conf.fold,
            transform=image_train(crop_size=conf.crop_size),
            crop_size=conf.crop_size,
            stage=1
        )
        train_loader = DataLoader(
            trainset,
            batch_size=conf.batch_size,
            shuffle=True,
            num_workers=conf.num_workers
        )
        valset = BP4D(
            conf.dataset_path,
            train=False,
            fold=conf.fold,
            transform=image_test(crop_size=conf.crop_size),
            stage=1
        )
        val_loader = DataLoader(
            valset,
            batch_size=conf.batch_size,
            shuffle=False,
            num_workers=conf.num_workers
        )

    elif conf.dataset == 'DISFA':
        trainset = DISFA(
            conf.dataset_path,
            train=True,
            fold=conf.fold,
            transform=image_train(crop_size=conf.crop_size),
            crop_size=conf.crop_size,
            stage=1
        )
        train_loader = DataLoader(
            trainset,
            batch_size=conf.batch_size,
            shuffle=True,
            num_workers=conf.num_workers
        )
        valset = DISFA(
            conf.dataset_path,
            train=False,
            fold=conf.fold,
            transform=image_test(crop_size=conf.crop_size),
            stage=1
        )
        val_loader = DataLoader(
            valset,
            batch_size=conf.batch_size,
            shuffle=False,
            num_workers=conf.num_workers
        )

    return train_loader, val_loader, len(trainset), len(valset)


# ============================================================
# 1. Pseudo-label Expression (giống Phase 1, dùng cho class center)
# ============================================================

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

    Y_a_float = Y_a.float()
    scores = Y_a_float @ M_AE  # (B, N_e)
    ke = scores.argmax(dim=1)  # (B,)

    neutral_mask = (Y_a_float.sum(dim=1) == 0)
    ke[neutral_mask] = neutral_index

    Y_e = torch.zeros(B, N_e, device=Y_a.device, dtype=Y_a_float.dtype)
    Y_e.scatter_(1, ke.unsqueeze(1), 1.0)
    return Y_e


# ma trận M_AE giống Phase 1
M_AE_np = np.load(r"matrixMAE\M_AE_DISFA.npy")
M_AE = torch.from_numpy(M_AE_np).float()


# ============================================================
# 2. CNF rule (prior knowledge) – chỉnh lại theo bảng FACS
# ============================================================

# 2.1. AU–Expression combo (CNF: ¬AU_i ∨ ... ∨ Emotion_k)
# 7 lớp expression: [Anger, Disg, Fear, Happ, Sad, Surp, Neutral]
# Dùng 6 emotion FACS (bỏ Contempt). Nếu emotion có ít AU trong DISFA
# thì dùng được bao nhiêu AU lấy bấy nhiêu.

# Angry,Fear,Happy,Sad,Surprise,Disgust,Neutral

CNF_AE_combo = [
    # Happy: 6 + 12
    ["¬AU6", "¬AU12", "Happy"],

    # Sadness: 1 + 4 (+15 nhưng DISFA không có AU15)
    ["¬AU1", "¬AU4", "Sad"],

    # Surprise: 1 + 2 + 5 + 26 (DISFA không có AU5 → 1,2,26)
    ["¬AU1", "¬AU2", "¬AU26", "Surprise"],

    # Fear: 1 + 2 + 4 + 5 + 7 + 20 + 26  (DISFA chỉ có 1,2,4,26)
    ["¬AU1", "¬AU2", "¬AU4", "¬AU26", "Fear"],

    # Anger: 4 + 5 + 7 + 23  (DISFA chỉ có AU4)
    ["¬AU4", "Angry"],

    # Disgust: 9 + 15 + 16  (DISFA chỉ có AU9)
    ["¬AU9", "Disgust"],
]


# 2.2. AU–AU Co-occurrence (AND semantics – dùng cho CNF sample-specific)
# Lấy tất cả cặp AU nằm trong cùng một combo emotion (sau khi rút gọn).

AU_AA_cooccur = [
    ["AU1", "AU2"],
    ["AU2", "AU4"],
    ["AU6", "AU12"],
    ["AU6", "AU25"],
    ["AU12", "AU25"],   # giữ 1 lần
    ["AU4", "AU9"],
    ["AU4", "AU25"],
    ["AU9", "AU25"],
    ["AU6", "AU26"],
    ["AU25", "AU26"],
]


# 2.3. AU–AU Mutual Exclusion (CNF: ¬AU_i ∨ ¬AU_j)
# Chọn các cặp hầu như không đi chung, và KHÔNG trùng với co-occur.

CNF_AA_exclusion = [
    ["¬AU2", "¬AU6"],   # data_exclusion
    ["¬AU2", "¬AU9"],   # data_exclusion
    ["¬AU1", "¬AU9"],   # rare + FACS conflict
    ["¬AU9", "¬AU12"],  # data_exclusion
    ["¬AU4", "¬AU12"],  # data_exclusion
]


# ============================================================
# 3. Tên AU / Emotion và mapping index
# ============================================================

# Thứ tự AU của DISFA (8 AU): 1,2,4,6,9,12,25,26
AU_NAMES = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]

AU_NAME_TO_IDX = {name: i for i, name in enumerate(AU_NAMES)}
IDX_TO_AU_NAME = {i: n for n, i in AU_NAME_TO_IDX.items()}

# Head emotion: [Anger, Disg, Fear, Happ, Sad, Surp, Neutral]
EXPR_NAME_TO_IDX = {
    "Angry":    0,
    "Fear":     1,
    "Happy":    2,
    "Sad":      3,
    "Surprise": 4,
    "Disgust":  5,
    # "Neutral": 6  # không cần cho rule
}
# Angry,Fear,Happy,Sad,Surprise,Disgust,Neutral

EXPR_NAMES = list(EXPR_NAME_TO_IDX.keys())
IDX_TO_EXPR_NAME = {j: n for n, j in EXPR_NAME_TO_IDX.items()}


# ============================================================
# 4. Tính class center cho AU / Expr từ Stage1 (Eq.(8))
# ============================================================

def compute_class_centers(
    conf,
    net_stage1,
    train_loader,
    num_aus,
    num_expr,
    neutral_index=6,
    alpha=0.1,   # hệ số EMA
):
    """
    Tính NC_a (AU centers) và NC_e (Expr centers) theo công thức (8) của SymGraphAU:
        nC_t = alpha * nC_{t-1} + (1 - alpha) * R_t   (t > 1)
        nC_1 = R_1
    """
    net_stage1.eval()
    device = next(net_stage1.parameters()).device

    centers_au = None      # (N_a, D)
    centers_expr = None    # (N_e, D)
    seen_au = None         # (N_a,) bool – đã có NC_1 cho AU_i chưa
    seen_expr = None       # (N_e,) bool

    print("==> Computing class centers (EMA, Eq.(8)) from Stage1...")
    from tqdm import tqdm

    with torch.no_grad():
        for inputs, targets in tqdm(train_loader, desc="Class centers (EMA)"):
            targets = targets.float().to(device)   # Y_a
            inputs = inputs.to(device)

            V_a, V_e, outputs_AU, outputs_Emo = net_stage1(inputs)
            B, Na, D = V_a.shape
            _, Ne, De = V_e.shape
            assert Na == num_aus, "num_aus mismatch"
            assert Ne == num_expr, "num_expr mismatch"
            assert D == De, "emb_dim mismatch"
            emb_dim = D

            if centers_au is None:
                centers_au = torch.zeros(Na, D, device=device)
                centers_expr = torch.zeros(Ne, D, device=device)
                seen_au = torch.zeros(Na, dtype=torch.bool, device=device)
                seen_expr = torch.zeros(Ne, dtype=torch.bool, device=device)

            # ===== AU centers =====
            for i in range(Na):
                mask = (targets[:, i] > 0.5)
                if mask.any():
                    R_t = V_a[mask, i, :].mean(dim=0)
                    if not seen_au[i]:
                        centers_au[i] = R_t
                        seen_au[i] = True
                    else:
                        centers_au[i] = alpha * centers_au[i] + (1.0 - alpha) * R_t

            # ===== Expr centers =====
            Y_e = au_to_expr_pseudo(targets, M_AE.to(device), neutral_index=neutral_index)
            expr_idx = Y_e.argmax(dim=1)  # (B,)

            for j in range(Ne):
                mask = (expr_idx == j)
                if mask.any():
                    R_t = V_e[mask, j, :].mean(dim=0)
                    if not seen_expr[j]:
                        centers_expr[j] = R_t
                        seen_expr[j] = True
                    else:
                        centers_expr[j] = alpha * centers_expr[j] + (1.0 - alpha) * R_t

    return centers_au.detach(), centers_expr.detach(), emb_dim


# ============================================================
# 5. GCN cơ bản (2-layer) + Discriminator + AND/OR emb
# ============================================================

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        out = torch.matmul(adj, support)
        if self.bias is not None:
            out = out + self.bias
        return out


class LogicGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(LogicGCN, self).__init__()
        self.gc1 = GraphConvolution(in_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, out_dim)

    def forward(self, x, adj):
        """
        x: (N, F_in)
        adj: (N, N), adjacency (0/1). Hàm này tự add self-loop + normalize.
        """
        device = x.device
        N = adj.size(0)

        adj_hat = adj + torch.eye(N, device=device)
        rowsum = adj_hat.sum(1)
        d_inv_sqrt = rowsum.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_norm = torch.matmul(torch.matmul(D_inv_sqrt, adj_hat), D_inv_sqrt)

        x = F.relu(self.gc1(x, adj_norm))
        x = F.relu(self.gc2(x, adj_norm))
        return x


class LogicDiscriminator(nn.Module):
    def __init__(self, emb_dim):
        super(LogicDiscriminator, self).__init__()
        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, z):
        return self.fc(z).squeeze(-1)


class LogicOperatorEmbeddings(nn.Module):
    def __init__(self, emb_dim):
        super(LogicOperatorEmbeddings, self).__init__()
        self.and_emb = nn.Parameter(torch.randn(emb_dim))
        self.or_emb = nn.Parameter(torch.randn(emb_dim))
        with torch.no_grad():
            self.and_emb.data = F.normalize(self.and_emb.data, dim=0)
            self.or_emb.data = F.normalize(self.or_emb.data, dim=0)


# ============================================================
# 6. RuleBase với PySAT – có p^uct + co-occur + filtered exclusion
# ============================================================

class LogicRuleBasePySAT:
    """
    - Giữ rule: CNF_AE_combo, CNF_AA_exclusion, AU_AA_cooccur
    - Xây CNF (PySAT) cho từng sample dựa trên y_a + p_expr
      + AE-clause thêm với xác suất p^uct lớn (uncertain)
      + Co-occurrence từ nhãn thật
      + Exclusion chỉ thêm khi cặp AU xuất hiện trong nhãn
    - Dùng Minisat22 để sinh assignment thỏa (G_s) và flip để sinh G_us
    """

    def __init__(self, num_aus, num_expr,
                 min_p_uct: float = 0.5,  # xác suất tối thiểu
                 max_p_uct: float = 0.95  # trần để "uncertain"
                 ):
        self.num_aus = num_aus
        self.num_expr = num_expr

        self.cnf_ae = CNF_AE_combo
        self.cnf_excl = CNF_AA_exclusion
        self.coccur_pairs = AU_AA_cooccur

        self.min_p_uct = min_p_uct
        self.max_p_uct = max_p_uct

        self.var2id = {}
        self.id2var = {}
        self._build_var_ids()

    def _build_var_ids(self):
        cur_id = 1
        # AU
        for name in AU_NAMES:
            if name not in self.var2id:
                self.var2id[name] = cur_id
                self.id2var[cur_id] = name
                cur_id += 1
        # Expr
        for name in EXPR_NAMES:
            if name not in self.var2id:
                self.var2id[name] = cur_id
                self.id2var[cur_id] = name
                cur_id += 1

    @staticmethod
    def _parse_literal_str(lit_str):
        lit_str = lit_str.strip()
        neg = lit_str.startswith("¬")
        if neg:
            name = lit_str[1:]
        else:
            name = lit_str
        return name, neg

    def _clause_str_to_int(self, clause_str_list):
        clause_int = []
        for lit in clause_str_list:
            name, neg = self._parse_literal_str(lit)
            if name not in self.var2id:
                continue
            vid = self.var2id[name]
            clause_int.append(-vid if neg else vid)
        return clause_int

    # ---------- xây CNF STR cho từng sample ----------

    def _build_cnf_str_for_sample(self, y_a, p_expr=None):
        """
        Trả về danh sách clause dạng string cho sample:
        - AE combo với xác suất p^uct (không phải luôn luôn)
        - Exclusion chỉ khi liên quan tới AU đang on trong nhãn
        - Co-occurrence từ nhãn thật
        - GT AU (unit clause)
        """
        y_a = y_a.detach().cpu().numpy()
        cnf_str = []

        # ----- AE combo với p^uct -----
        for clause in self.cnf_ae:
            emo_name, emo_neg = self._parse_literal_str(clause[-1])
            if emo_name in EXPR_NAME_TO_IDX and not emo_neg and p_expr is not None:
                j = EXPR_NAME_TO_IDX[emo_name]
                p_e = float(p_expr[j].item())
                p_uct = max(self.min_p_uct, min(self.max_p_uct, p_e))
            else:
                p_uct = 0.8

            if random.random() < p_uct:
                cnf_str.append(clause)

        # ----- Exclusion: chỉ thêm khi cặp xuất hiện trong nhãn -----
        for clause in self.cnf_excl:
            if len(clause) != 2:
                continue
            name1, _ = self._parse_literal_str(clause[0])
            name2, _ = self._parse_literal_str(clause[1])
            if name1 in AU_NAME_TO_IDX and name2 in AU_NAME_TO_IDX:
                i = AU_NAME_TO_IDX[name1]
                j = AU_NAME_TO_IDX[name2]
                if i < len(y_a) and j < len(y_a):
                    if (y_a[i] > 0.5) or (y_a[j] > 0.5):
                        cnf_str.append(clause)
                else:
                    cnf_str.append(clause)
            else:
                cnf_str.append(clause)

        # ----- Co-occurrence từ nhãn thật -----
        for pair in self.coccur_pairs:
            if len(pair) != 2:
                continue
            au1, au2 = pair
            if au1 in AU_NAME_TO_IDX and au2 in AU_NAME_TO_IDX:
                i = AU_NAME_TO_IDX[au1]
                j = AU_NAME_TO_IDX[au2]
                if i < len(y_a) and j < len(y_a):
                    if (y_a[i] > 0.5) and (y_a[j] > 0.5):
                        cnf_str.append([f"¬{au1}", au2])
                        cnf_str.append([f"¬{au2}", au1])

        # ----- GT AU = 1 → unit clause -----
        for idx, val in enumerate(y_a):
            if idx >= len(AU_NAMES):
                continue
            au_name = AU_NAMES[idx]
            if val > 0.5:
                cnf_str.append([au_name])

        return cnf_str

    def _build_base_cnf_for_sample(self, y_a, p_expr=None):
        cnf_str = self._build_cnf_str_for_sample(y_a, p_expr)
        cnf = CNF()
        for clause in cnf_str:
            ints = self._clause_str_to_int(clause)
            if len(ints) > 0:
                cnf.append(ints)
        return cnf, cnf_str

    # ---------- sinh assignment ----------

    def sample_satisfying_assignment(self, y_a, p_expr=None):
        """
        Dùng PySAT để tìm assignment thỏa CNF
        Trả về:
            assign_s: {('au', i): 0/1, ('expr', j): 0/1}
            cnf_str: list clause string của sample đó
        """
        cnf, cnf_str = self._build_base_cnf_for_sample(y_a, p_expr)
        with Minisat22(bootstrap_with=cnf) as solver:
            sat = solver.solve()
            if not sat:
                assign_s = {}
                for i in range(self.num_aus):
                    v = float(y_a[i].item()) if i < y_a.shape[0] else 0.0
                    assign_s[('au', i)] = v
                for j in range(self.num_expr):
                    assign_s[('expr', j)] = 0.0
                return assign_s, cnf_str
            model = solver.get_model()

        assign_s = {}
        for i in range(self.num_aus):
            assign_s[('au', i)] = 0.0
        for j in range(self.num_expr):
            assign_s[('expr', j)] = 0.0

        for lit in model:
            v = abs(lit)
            val = 1.0 if lit > 0 else 0.0
            if v not in self.id2var:
                continue
            name = self.id2var[v]
            if name in AU_NAME_TO_IDX:
                i = AU_NAME_TO_IDX[name]
                if i < self.num_aus:
                    assign_s[('au', i)] = val
            elif name in EXPR_NAME_TO_IDX:
                j = EXPR_NAME_TO_IDX[name]
                if j < self.num_expr:
                    assign_s[('expr', j)] = val

        return assign_s, cnf_str

    # ---------- evaluate CNF string ----------

    def eval_clause(self, clause, assign):
        for lit in clause:
            name, neg = self._parse_literal_str(lit)
            val = 0.0
            if name in AU_NAME_TO_IDX:
                i = AU_NAME_TO_IDX[name]
                val = assign.get(('au', i), 0.0)
            elif name in EXPR_NAME_TO_IDX:
                j = EXPR_NAME_TO_IDX[name]
                val = assign.get(('expr', j), 0.0)

            if (not neg and val > 0.5) or (neg and val < 0.5):
                return True
        return False

    def eval_cnf(self, cnf_str, assign):
        for clause in cnf_str:
            if not self.eval_clause(clause, assign):
                return False
        return True

    def sample_assignments_with_pysat(self, y_a, p_expr=None):
        """
        Trả về:
            assign_s: assignment thỏa CNF
            assign_us: assignment vi phạm CNF ít nhất 1 clause
            cnf_str: CNF string paradigm của sample
        """
        assign_s, cnf_str = self.sample_satisfying_assignment(y_a, p_expr)

        assign_us = dict(assign_s)
        var_keys = list(assign_us.keys())
        max_try = 20

        for _ in range(max_try):
            k = random.choice(var_keys)
            assign_us[k] = 1.0 - assign_us[k]
            if self.eval_cnf(cnf_str, assign_us):
                continue
            else:
                return assign_s, assign_us, cnf_str

        # fallback: flip tất cả AU
        for k in var_keys:
            if k[0] == 'au':
                assign_us[k] = 1.0 - assign_us[k]
        return assign_s, assign_us, cnf_str


# ============================================================
# 7. Xây logic graph từ CNF string + centers + global
# ============================================================

NODE_TYPE_GLOBAL = 0
NODE_TYPE_AND = 1
NODE_TYPE_OR = 2
NODE_TYPE_LEAF = 3  # AU / Expr

def make_type_code(node_type, device):
    code = torch.zeros(4, device=device)
    code[node_type] = 1.0
    return code

def parse_literal_str(lit_str):
    lit_str = lit_str.strip()
    neg = lit_str.startswith("¬")
    if neg:
        name = lit_str[1:]
    else:
        name = lit_str
    return name, neg

def build_logic_graph_for_sample(
    cnf_clauses_str,  # list[clause], mỗi clause = list[str]
    centers_au,      # (N_a, D)
    centers_expr,    # (N_e, D)
    global_vec,      # (D,)
    op_emb_module,   # LogicOperatorEmbeddings
    emb_dim,
    assignment=None, # dict{('au',i): scalar, ('expr',j): scalar} hoặc None
    device='cuda'
):
    num_aus = centers_au.size(0)
    num_expr = centers_expr.size(0)
    num_clause = len(cnf_clauses_str)

    # chỉ số node
    idx_au = {i: i for i in range(num_aus)}
    idx_expr = {j: num_aus + j for j in range(num_expr)}
    offset_or = num_aus + num_expr
    idx_or = {c: offset_or + c for c in range(num_clause)}
    idx_and = offset_or + num_clause
    idx_global = idx_and + 1

    num_nodes = idx_global + 1
    A = torch.zeros(num_nodes, num_nodes, device=device)

    # OR node kết nối AND + literal
    for c_idx, clause in enumerate(cnf_clauses_str):
        or_idx = idx_or[c_idx]
        A[or_idx, idx_and] = 1.0
        A[idx_and, or_idx] = 1.0
        for lit in clause:
            name, neg = parse_literal_str(lit)
            if name in AU_NAME_TO_IDX:
                i = AU_NAME_TO_IDX[name]
                if i >= num_aus:
                    continue
                lit_idx = idx_au[i]
            elif name in EXPR_NAME_TO_IDX:
                j = EXPR_NAME_TO_IDX[name]
                if j >= num_expr:
                    continue
                lit_idx = idx_expr[j]
            else:
                continue
            A[or_idx, lit_idx] = 1.0
            A[lit_idx, or_idx] = 1.0

    # Global nối với tất cả
    for n in range(num_nodes):
        if n == idx_global:
            continue
        A[idx_global, n] = 1.0
        A[n, idx_global] = 1.0

    X = torch.zeros(num_nodes, emb_dim + 4, device=device)

    # AU leaf
    for i in range(num_aus):
        node_idx = idx_au[i]
        type_code = make_type_code(NODE_TYPE_LEAF, device)
        base_vec = centers_au[i]
        if assignment is not None:
            scale = assignment.get(('au', i), 0.0)
            sem_vec = scale * base_vec
        else:
            sem_vec = base_vec
        X[node_idx, :4] = type_code
        X[node_idx, 4:] = sem_vec

    # Expr leaf
    for j in range(num_expr):
        node_idx = idx_expr[j]
        type_code = make_type_code(NODE_TYPE_LEAF, device)
        base_vec = centers_expr[j]
        if assignment is not None:
            scale = assignment.get(('expr', j), 0.0)
            sem_vec = scale * base_vec
        else:
            sem_vec = base_vec
        X[node_idx, :4] = type_code
        X[node_idx, 4:] = sem_vec

    # OR clause
    for c_idx in range(num_clause):
        node_idx = idx_or[c_idx]
        type_code = make_type_code(NODE_TYPE_OR, device)
        X[node_idx, :4] = type_code
        X[node_idx, 4:] = op_emb_module.or_emb

    # AND
    type_code_and = make_type_code(NODE_TYPE_AND, device)
    X[idx_and, :4] = type_code_and
    X[idx_and, 4:] = op_emb_module.and_emb

    # Global
    type_code_global = make_type_code(NODE_TYPE_GLOBAL, device)
    X[idx_global, :4] = type_code_global
    X[idx_global, 4:] = global_vec

    return X, A, idx_global


# ============================================================
# 8. Loss Phase 2: Triplet margin + BCE
# ============================================================

def logic_triplet_loss(z_cnf, z_s, z_us, margin=1.0):
    """
    Triplet loss kiểu chuẩn:
        L = mean( max(0, margin + d_s - d_us) )
    d_s  = ||z_cnf - z_s||^2
    d_us = ||z_cnf - z_us||^2
    """
    d_s = ((z_cnf - z_s) ** 2).sum(dim=1)
    d_us = ((z_cnf - z_us) ** 2).sum(dim=1)
    return F.relu(margin + d_s - d_us).mean()


# ============================================================
# 9. Train / Val Phase 2
# ============================================================

def train_phase2(
    conf,
    net_stage1,
    train_loader,
    gcn,
    disc,
    op_emb_module,
    centers_au,
    centers_expr,
    emb_dim,
    rule_base_pysat,
    optimizer,
    epoch,
    lambda_c=0.1,
    margin_h=1.0,
):
    net_stage1.eval()
    gcn.train()
    disc.train()
    op_emb_module.train()

    device = next(gcn.parameters()).device
    losses_total = AverageMeter()
    losses_h = AverageMeter()
    losses_c = AverageMeter()
    acc_disc_meter = AverageMeter()
    triplet_sat_meter = AverageMeter()

    pbar = tqdm(train_loader, desc=f"[Phase2 Train] Epoch {epoch}")

    for batch_idx, (inputs, targets) in enumerate(pbar):
        targets = targets.float()
        if torch.cuda.is_available():
            inputs = inputs.to(device)
            targets = targets.to(device)

        B = inputs.size(0)
        optimizer.zero_grad()

        with torch.no_grad():
            V_a, V_e, outputs_AU, outputs_Emo = net_stage1(inputs)
            global_vec_batch = V_a.mean(dim=1)
            p_expr_batch = torch.sigmoid(outputs_Emo)

        z_cnf_list, z_s_list, z_us_list = [], [], []

        for b in range(B):
            y_a_b = targets[b]
            global_vec_b = global_vec_batch[b]
            p_expr_b = p_expr_batch[b]

            assign_s, assign_us, cnf_str_b = rule_base_pysat.sample_assignments_with_pysat(
                y_a_b, p_expr_b
            )

            X_cnf, A_cnf, idx_g = build_logic_graph_for_sample(
                cnf_str_b, centers_au, centers_expr, global_vec_b,
                op_emb_module, emb_dim, assignment=None, device=device
            )
            X_s, A_s, idx_g_s = build_logic_graph_for_sample(
                cnf_str_b, centers_au, centers_expr, global_vec_b,
                op_emb_module, emb_dim, assignment=assign_s, device=device
            )
            X_us, A_us, idx_g_us = build_logic_graph_for_sample(
                cnf_str_b, centers_au, centers_expr, global_vec_b,
                op_emb_module, emb_dim, assignment=assign_us, device=device
            )

            Q_cnf = gcn(X_cnf, A_cnf)
            Q_s = gcn(X_s, A_s)
            Q_us = gcn(X_us, A_us)

            z_cnf_list.append(Q_cnf[idx_g])
            z_s_list.append(Q_s[idx_g_s])
            z_us_list.append(Q_us[idx_g_us])

        z_cnf = torch.stack(z_cnf_list, dim=0)
        z_s = torch.stack(z_s_list, dim=0)
        z_us = torch.stack(z_us_list, dim=0)

        # ----- L_h -----
        L_h = logic_triplet_loss(z_cnf, z_s, z_us, margin=margin_h)

        # ----- L_c (disc) -----
        emb_logic = torch.cat([z_s, z_us], dim=0)
        labels_logic = torch.cat([
            torch.ones(B, device=device),
            torch.zeros(B, device=device)
        ], dim=0)
        logits_logic = disc(emb_logic)
        L_c = F.binary_cross_entropy_with_logits(logits_logic, labels_logic)

        loss = L_h + lambda_c * L_c
        loss.backward()
        optimizer.step()

        # ----- metrics -----
        with torch.no_grad():
            prob = torch.sigmoid(logits_logic)
            pred = (prob > 0.5).float()
            acc_disc = (pred == labels_logic).float().mean().item()

            d_s = ((z_cnf - z_s) ** 2).sum(dim=1)
            d_us = ((z_cnf - z_us) ** 2).sum(dim=1)
            triplet_sat = (d_us > d_s + margin_h).float().mean().item()

        losses_total.update(loss.item(), B)
        losses_h.update(L_h.item(), B)
        losses_c.update(L_c.item(), B)
        acc_disc_meter.update(acc_disc, B)
        triplet_sat_meter.update(triplet_sat, B)

        pbar.set_postfix({
            'L_total': f"{loss.item():.4f}",
            'L_h': f"{L_h.item():.4f}",
            'L_c': f"{L_c.item():.4f}",
            'acc_disc': f"{acc_disc_meter.avg:.3f}",
            'triplet@margin': f"{triplet_sat_meter.avg:.3f}",
        })

    return (
        losses_total.avg,
        losses_h.avg,
        losses_c.avg,
        acc_disc_meter.avg,
        triplet_sat_meter.avg,
    )


@torch.no_grad()
def val_phase2(
    conf,
    net_stage1,
    val_loader,
    gcn,
    disc,
    op_emb_module,
    centers_au,
    centers_expr,
    emb_dim,
    rule_base_pysat,
    lambda_c=0.1,
    margin_h=1.0,
):
    net_stage1.eval()
    gcn.eval()
    disc.eval()
    op_emb_module.eval()

    device = next(gcn.parameters()).device
    losses_total = AverageMeter()
    losses_h = AverageMeter()
    losses_c = AverageMeter()
    acc_disc_meter = AverageMeter()
    triplet_sat_meter = AverageMeter()

    pbar = tqdm(val_loader, desc=f"[Phase2 Val]")

    for batch_idx, (inputs, targets) in enumerate(pbar):
        targets = targets.float()
        if torch.cuda.is_available():
            inputs = inputs.to(device)
            targets = targets.to(device)

        B = inputs.size(0)

        V_a, V_e, outputs_AU, outputs_Emo = net_stage1(inputs)
        global_vec_batch = V_a.mean(dim=1)
        p_expr_batch = torch.sigmoid(outputs_Emo)

        z_cnf_list, z_s_list, z_us_list = [], [], []

        for b in range(B):
            y_a_b = targets[b]
            global_vec_b = global_vec_batch[b]
            p_expr_b = p_expr_batch[b]

            assign_s, assign_us, cnf_str_b = rule_base_pysat.sample_assignments_with_pysat(
                y_a_b, p_expr_b
            )

            X_cnf, A_cnf, idx_g = build_logic_graph_for_sample(
                cnf_str_b, centers_au, centers_expr, global_vec_b,
                op_emb_module, emb_dim, assignment=None, device=device
            )
            X_s, A_s, idx_g_s = build_logic_graph_for_sample(
                cnf_str_b, centers_au, centers_expr, global_vec_b,
                op_emb_module, emb_dim, assignment=assign_s, device=device
            )
            X_us, A_us, idx_g_us = build_logic_graph_for_sample(
                cnf_str_b, centers_au, centers_expr, global_vec_b,
                op_emb_module, emb_dim, assignment=assign_us, device=device
            )

            Q_cnf = gcn(X_cnf, A_cnf)
            Q_s = gcn(X_s, A_s)
            Q_us = gcn(X_us, A_us)

            z_cnf_list.append(Q_cnf[idx_g])
            z_s_list.append(Q_s[idx_g_s])
            z_us_list.append(Q_us[idx_g_us])

        z_cnf = torch.stack(z_cnf_list, dim=0)
        z_s = torch.stack(z_s_list, dim=0)
        z_us = torch.stack(z_us_list, dim=0)

        # ----- L_h -----
        L_h = logic_triplet_loss(z_cnf, z_s, z_us, margin=margin_h)

        # ----- L_c -----
        emb_logic = torch.cat([z_s, z_us], dim=0)
        labels_logic = torch.cat([
            torch.ones(B, device=device),
            torch.zeros(B, device=device)
        ], dim=0)
        logits_logic = disc(emb_logic)
        L_c = F.binary_cross_entropy_with_logits(logits_logic, labels_logic)

        loss = L_h + lambda_c * L_c

        prob = torch.sigmoid(logits_logic)
        pred = (prob > 0.5).float()
        acc_disc = (pred == labels_logic).float().mean().item()

        d_s = ((z_cnf - z_s) ** 2).sum(dim=1)
        d_us = ((z_cnf - z_us) ** 2).sum(dim=1)
        triplet_sat = (d_us > d_s + margin_h).float().mean().item()

        losses_total.update(loss.item(), B)
        losses_h.update(L_h.item(), B)
        losses_c.update(L_c.item(), B)
        acc_disc_meter.update(acc_disc, B)
        triplet_sat_meter.update(triplet_sat, B)

        pbar.set_postfix({
            'L_total': f"{loss.item():.4f}",
            'L_h': f"{L_h.item():.4f}",
            'L_c': f"{L_c.item():.4f}",
            'acc_disc': f"{acc_disc_meter.avg:.3f}",
            'triplet@margin': f"{triplet_sat_meter.avg:.3f}",
        })

    return (
        losses_total.avg,
        losses_h.avg,
        losses_c.avg,
        acc_disc_meter.avg,
        triplet_sat_meter.avg,
    )


# ============================================================
# 10. main() cho Phase 2
# ============================================================
def compute_au_pair_stats(train_loader, device):
    """
    Quét toàn bộ nhãn AU trong train_loader để thống kê:
      - single_cnt[i]: số frame có AU_i = 1
      - co_cnt[(i,j)]: số frame có AU_i = 1 và AU_j = 1
      - total_frames: tổng số frame
    """
    single_cnt = {}    # i -> count
    co_cnt = {}        # (i,j) với i<j -> count
    total_frames = 0

    for inputs, targets in tqdm(train_loader, desc="Scan AU pairs"):
        y = targets.float().to(device)   # (B, N_a)
        y_bin = (y > 0.5).int()         # nhị phân 0/1

        B, Na = y_bin.shape
        total_frames += B

        for b in range(B):
            on_idx = torch.nonzero(y_bin[b]).view(-1).tolist()  # danh sách AU đang bật

            # đếm từng AU riêng lẻ
            for i in on_idx:
                single_cnt[i] = single_cnt.get(i, 0) + 1

            # đếm cặp (i,j) cùng bật
            for u in range(len(on_idx)):
                for v in range(u + 1, len(on_idx)):
                    a, b2 = sorted([on_idx[u], on_idx[v]])
                    key = (a, b2)
                    co_cnt[key] = co_cnt.get(key, 0) + 1

    return single_cnt, co_cnt, total_frames

def export_au_pair_stats_to_csv(
    train_loader,
    out_csv_path="au_pair_stats_DISFA.csv",
    out_txt_path=None,
    cooccur_pairs=None,
    excl_clauses=None,
):
    """
    - Quét train_loader DISFA → thống kê AU pairs
    - Gộp thông tin lý thuyết (AU_AA_cooccur, CNF_AA_exclusion)
    - Xuất ra CSV + TXT
    - Thêm:
        + data_only_pairs: data có mà luật không có
        + rule_only_pairs: luật có mà data không có
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    single_cnt, co_cnt, total_frames = compute_au_pair_stats(train_loader, device)

    # Chuẩn bị map nhanh từ index -> tên AU
    idx_to_name = {i: n for i, n in enumerate(AU_NAMES)}

    # Chuẩn bị set cặp lý thuyết
    cooccur_pairs = cooccur_pairs or AU_AA_cooccur
    excl_clauses = excl_clauses or CNF_AA_exclusion

    co_theory = set()
    for au1, au2 in cooccur_pairs:
        i, j = AU_NAME_TO_IDX[au1], AU_NAME_TO_IDX[au2]
        if i > j:
            i, j = j, i
        co_theory.add((i, j))

    excl_theory = set()
    for clause in excl_clauses:
        # clause dạng ["¬AU2", "¬AU4"]
        name1 = clause[0].replace("¬", "")
        name2 = clause[1].replace("¬", "")
        i, j = AU_NAME_TO_IDX[name1], AU_NAME_TO_IDX[name2]
        if i > j:
            i, j = j, i
        excl_theory.add((i, j))

    # Tập cặp xuất hiện trong data (ít nhất 1 lần)
    data_pairs = set(k for k, v in co_cnt.items() if v > 0)

    # Tập tất cả cặp cần report (union data + lý thuyết)
    all_pairs = set(data_pairs)
    for p in co_theory.union(excl_theory):
        all_pairs.add(p)

    lines_txt = []

    # ---- Xuất CSV ----
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "au1",
            "au2",
            "count_au1",
            "count_au2",
            "count_both",
            "p_both_over_frames",
            "p_both_given_au1",
            "p_both_given_au2",
            "rule_theory",   # cooccur / exclusion / co+ex / none
            "data_tag"       # data_cooccur / data_exclusion / ambiguous
        ])

        for (i, j) in sorted(all_pairs):
            au1 = idx_to_name.get(i, f"AU{i}")
            au2 = idx_to_name.get(j, f"AU{j}")

            c1 = single_cnt.get(i, 0)
            c2 = single_cnt.get(j, 0)
            cboth = co_cnt.get((i, j), 0)

            # Tỉ lệ
            p_both_over_frames = cboth / total_frames if total_frames > 0 else 0.0
            p_both_given_au1 = cboth / c1 if c1 > 0 else 0.0
            p_both_given_au2 = cboth / c2 if c2 > 0 else 0.0

            # Loại luật lý thuyết
            is_co = (i, j) in co_theory
            is_ex = (i, j) in excl_theory
            if is_co and is_ex:
                rule_theory = "co+ex"
            elif is_co:
                rule_theory = "cooccur"
            elif is_ex:
                rule_theory = "exclusion"
            else:
                rule_theory = "none"

            # Tag theo data (bạn chỉnh threshold nếu muốn)
            max_cond = max(p_both_given_au1, p_both_given_au2)
            if max_cond > 0.3:
                data_tag = "data_cooccur"
            elif max_cond < 0.05:
                data_tag = "data_exclusion"
            else:
                data_tag = "ambiguous"

            writer.writerow([
                au1,
                au2,
                c1,
                c2,
                cboth,
                f"{p_both_over_frames:.6f}",
                f"{p_both_given_au1:.6f}",
                f"{p_both_given_au2:.6f}",
                rule_theory,
                data_tag
            ])

            if out_txt_path is not None:
                lines_txt.append(
                    f"{au1}-{au2}: "
                    f"count_au1={c1}, count_au2={c2}, count_both={cboth}, "
                    f"p_both|au1={p_both_given_au1:.3f}, "
                    f"p_both|au2={p_both_given_au2:.3f}, "
                    f"theory={rule_theory}, data={data_tag}"
                )

    # ---- Tính thêm 2 list đặc biệt ----
    # 1) data_only_pairs: data có mà luật không có
    data_only_pairs = [
        p for p in data_pairs
        if (p not in co_theory) and (p not in excl_theory)
    ]

    # 2) rule_only_pairs: luật có mà data không có
    rule_only_pairs = [
        p for p in co_theory.union(excl_theory)
        if co_cnt.get(p, 0) == 0
    ]

    if out_txt_path is not None:
        with open(out_txt_path, "w", encoding="utf-8") as ftxt:
            # Phần chi tiết từng cặp
            ftxt.write("=== TẤT CẢ CẶP AU (theory + data) ===\n")
            ftxt.write("\n".join(lines_txt))
            ftxt.write("\n\n")

            # In data_only_pairs
            ftxt.write("=== DATA_ONLY_PAIRS (data có, luật không có) ===\n")
            for (i, j) in sorted(data_only_pairs):
                au1 = idx_to_name.get(i, f"AU{i}")
                au2 = idx_to_name.get(j, f"AU{j}")
                cboth = co_cnt.get((i, j), 0)
                c1 = single_cnt.get(i, 0)
                c2 = single_cnt.get(j, 0)
                ftxt.write(
                    f"{au1}-{au2}: count_both={cboth}, "
                    f"count_au1={c1}, count_au2={c2}\n"
                )

            ftxt.write("\n=== RULE_ONLY_PAIRS (luật có, data không có) ===\n")
            for (i, j) in sorted(rule_only_pairs):
                au1 = idx_to_name.get(i, f"AU{i}")
                au2 = idx_to_name.get(j, f"AU{j}")
                # xem thuộc luật nào
                tag = []
                if (i, j) in co_theory:
                    tag.append("cooccur")
                if (i, j) in excl_theory:
                    tag.append("exclusion")
                tag_str = "+".join(tag) if tag else "none"
                ftxt.write(f"{au1}-{au2}: theory={tag_str}, count_both=0\n")

    print(f"[SAVED] AU pair stats → {out_csv_path}")
    if out_txt_path is not None:
        print(f"[SAVED] Human-readable AU pair stats → {out_txt_path}")
        print("[INFO] data_only_pairs =", len(data_only_pairs),
              "| rule_only_pairs =", len(rule_only_pairs))

def main(conf):

    if conf.dataset == 'BP4D':
        dataset_info = BP4D_infolist
    elif conf.dataset == 'DISFA':
        dataset_info = DISFA_infolist

    # ---------- Dataloader ----------
    train_loader, val_loader, train_data_num, val_data_num = get_dataloader(conf)
    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold, conf.N_fold, val_data_num))


    # ====== DEBUG: Xuất thống kê AU pair trước khi train Phase2 ======
    export_au_pair_stats_to_csv(
        train_loader,
        out_csv_path=f"au_pair_stats_{conf.dataset}_fold{conf.fold}.csv",
        out_txt_path=f"au_pair_stats_{conf.dataset}_fold{conf.fold}.txt",
        cooccur_pairs=AU_AA_cooccur,
        excl_clauses=CNF_AA_exclusion,
    )

    print('export_au_pair_stats_to_csv', f"au_pair_stats_{conf.dataset}_fold{conf.fold}.csv" )

    # ---------- Stage 1 model (đã train xong Phase 1) ----------
    net_stage1 = MEFARGStage1(num_aus=conf.num_classes, backbone=conf.arc, num_expr=7)

    assert conf.resume != '', "For Phase 2, conf.resume must be Stage1 checkpoint path"
    logging.info("Resume Stage1 from | {} ]".format(conf.resume))
    net_stage1 = load_state_dict(net_stage1, conf.resume)

    if torch.cuda.is_available():
        net_stage1 = nn.DataParallel(net_stage1).cuda()
        global M_AE
        M_AE = M_AE.cuda()

    # Đóng băng Stage 1 trong Phase 2
    for p in net_stage1.parameters():
        p.requires_grad = False

    # ---------- Tính class centers (Eq.(8)) ----------
    centers_au, centers_expr, emb_dim = compute_class_centers(
        conf,
        net_stage1,
        train_loader,
        num_aus=conf.num_classes,
        num_expr=7,
        neutral_index=6
    )
    if torch.cuda.is_available():
        centers_au = centers_au.cuda()
        centers_expr = centers_expr.cuda()

    # ---------- Logic modules (GCN + Disc + AND/OR emb) ----------
    in_dim = emb_dim + 4          # 4 dim type-code + D dim semantic
    hidden_dim = emb_dim
    out_dim = emb_dim
    gcn = LogicGCN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    disc = LogicDiscriminator(emb_dim=emb_dim)
    op_emb_module = LogicOperatorEmbeddings(emb_dim=emb_dim)

    if torch.cuda.is_available():
        gcn = gcn.cuda()
        disc = disc.cuda()
        op_emb_module = op_emb_module.cuda()

    # ---------- Rule base với p^uct + co-occ + filtered exclusion ----------
    rule_base_pysat = LogicRuleBasePySAT(num_aus=conf.num_classes, num_expr=7)

    # ---------- Optimizer ----------
    optimizer = optim.AdamW(
        list(gcn.parameters()) +
        list(disc.parameters()) +
        list(op_emb_module.parameters()),
        betas=(0.9, 0.999),
        lr=conf.learning_rate,
        weight_decay=conf.weight_decay
    )

    print('Phase 2 init learning rate:', conf.learning_rate)
    start_time = datetime.now()
    print("Start time:", start_time.strftime("%Y-%m-%d %H:%M:%S"))

    lambda_c = 0.1
    margin_h = 1.0

    # ---------- Training loop ----------
    for epoch in range(conf.epochs):
        save_path = os.path.join(
            conf['outdir'],
            f'phase2_epoch{epoch+1}_fold{conf.fold}.pth'
        )
        print("--Phase2 Weights will be saved at", save_path)

        logging.info(
            f"[Phase2 CFG] arc={conf.arc} K={conf.neighbor_num} "
            f"metric={conf.metric} bs={conf.batch_size} "
            f"lr0={conf.learning_rate} wd={conf.weight_decay}"
        )

        lr = optimizer.param_groups[0]['lr']
        logging.info(
            "Phase2 Epoch: [{} | {} LR: {} ]".format(
                epoch + 1, conf.epochs, lr
            )
        )

        # ----- Train -----
        (train_L_total,
         train_L_h,
         train_L_c,
         train_acc_disc,
         train_triplet_sat) = train_phase2(
            conf,
            net_stage1,
            train_loader,
            gcn,
            disc,
            op_emb_module,
            centers_au,
            centers_expr,
            emb_dim,
            rule_base_pysat,
            optimizer,
            epoch,
            lambda_c=lambda_c,
            margin_h=margin_h,
        )

        # ----- Val -----
        (val_L_total,
         val_L_h,
         val_L_c,
         val_acc_disc,
         val_triplet_sat) = val_phase2(
            conf,
            net_stage1,
            val_loader,
            gcn,
            disc,
            op_emb_module,
            centers_au,
            centers_expr,
            emb_dim,
            rule_base_pysat,
            lambda_c=lambda_c,
            margin_h=margin_h,
        )

        # ----- Log -----
        infostr = {
            'Phase2 Epoch: {}  train_L_total: {:.5f}  val_L_total: {:.5f}  '
            'train_L_h: {:.5f}  train_L_c: {:.5f}  val_L_h: {:.5f}  val_L_c: {:.5f}  '
            'train_acc_disc: {:.3f}  val_acc_disc: {:.3f}  '
            'train_triplet@margin: {:.3f}  val_triplet@margin: {:.3f}'.format(
                epoch + 1, train_L_total, val_L_total,
                train_L_h, train_L_c, val_L_h, val_L_c,
                train_acc_disc, val_acc_disc,
                train_triplet_sat, val_triplet_sat,
            )
        }
        print(infostr)
        logging.info(str(infostr))

        # ----- Save checkpoint -----
        if (epoch + 1) % 2 == 0:
            checkpoint = {
                'epoch': epoch,
                'gcn_state_dict': gcn.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'op_emb_state_dict': op_emb_module.state_dict(),
                'centers_au': centers_au,
                'centers_expr': centers_expr,
            }
            torch.save(checkpoint, save_path)

    end_time = datetime.now()
    print("End time:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Duration:", end_time - start_time)




# ============================================================
# 11. Run
# ============================================================

if __name__ == "__main__":
    conf = get_config()
    set_env(conf)
    set_outdir(conf)
    set_logger(conf)
    main(conf)
