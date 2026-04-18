import torch
import torch.nn as nn
import numpy as np


# ── Cạnh topology dlib 68-pt ─────────────────────────────────────────────────

_EDGES = [
    # Jawline 0-16
    *[(i, i + 1) for i in range(0, 16)],
    # Left brow 17-21
    *[(i, i + 1) for i in range(17, 21)],
    # Right brow 22-26
    *[(i, i + 1) for i in range(22, 26)],
    # Nose bridge 27-30
    *[(i, i + 1) for i in range(27, 30)],
    # Nose lower 30-35
    *[(i, i + 1) for i in range(30, 35)], (30, 35),
    # Left eye 36-41 (vòng)
    *[(i, i + 1) for i in range(36, 41)], (41, 36),
    # Right eye 42-47 (vòng)
    *[(i, i + 1) for i in range(42, 47)], (47, 42),
    # Outer mouth 48-59 (vòng)
    *[(i, i + 1) for i in range(48, 59)], (59, 48),
    # Inner mouth 60-67 (vòng)
    *[(i, i + 1) for i in range(60, 67)], (67, 60),
    # Cross: brow ↔ eye
    (17, 36), (21, 39), (22, 42), (26, 45),
    # Cross: nose ↔ mouth
    (30, 48), (30, 51), (30, 57),
    # Jaw ↔ mouth
    (8, 57),
]

_N = 68  # số landmarks


def _build_adj() -> torch.Tensor:
    """Trả về A_hat = D^{-1/2}(A+I)D^{-1/2}, shape (68,68), float32."""
    A = np.zeros((_N, _N), dtype=np.float32)
    for i, j in _EDGES:
        A[i, j] = A[j, i] = 1.0
    A += np.eye(_N, dtype=np.float32)          # self-loop
    d = A.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-6)))
    return torch.from_numpy(d_inv_sqrt @ A @ d_inv_sqrt)


# Pre-compute một lần khi import
_ADJ = _build_adj()   # (68, 68)


# ── AU → landmark index mapping (DISFA 8 AUs) ────────────────────────────────

AU_LANDMARK_MAP = {
    'AU1':  list(range(17, 27)),   # Inner brow raiser  → lông mày
    'AU2':  list(range(17, 27)),   # Outer brow raiser  → lông mày
    'AU4':  list(range(17, 27)),   # Brow lowerer       → lông mày
    'AU6':  list(range(36, 48)),   # Cheek raiser       → mắt
    'AU9':  list(range(27, 36)),   # Nose wrinkler      → mũi
    'AU12': list(range(48, 68)),   # Lip corner puller  → miệng
    'AU25': list(range(48, 68)),   # Lips part          → miệng
    'AU26': list(range(48, 68)),   # Jaw drop           → miệng
}

DISFA_AU_ORDER = ['AU1', 'AU2', 'AU4', 'AU6', 'AU9', 'AU12', 'AU25', 'AU26']

# Pre-compute list of index lists theo thứ tự DISFA
_AU_POOL_IDX = [AU_LANDMARK_MAP[au] for au in DISFA_AU_ORDER]


# ── GCNLayer ──────────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    """H' = ReLU(BN(A_hat · H · W))   –   (B,68,Cin) → (B,68,Cout)"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch, bias=False)
        self.bn     = nn.BatchNorm1d(_N)
        self.act    = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # adj: (68,68), x: (B,68,Cin)
        h = torch.matmul(adj.unsqueeze(0), x)   # (B,68,Cin)
        h = self.linear(h)                       # (B,68,Cout)
        h = self.bn(h)
        return self.act(h)


# ── LandmarkGCN ───────────────────────────────────────────────────────────────

class LandmarkGCN(nn.Module):

    def __init__(self, in_channels: int = 2, gcn_hidden: tuple = (64, 128)):
        super().__init__()
        layers, prev = [], in_channels
        for ch in gcn_hidden:
            layers.append(GCNLayer(prev, ch))
            prev = ch
        self.layers      = nn.ModuleList(layers)
        self.out_channels = prev
        self.register_buffer('adj', _ADJ)   # không phải parameter

    def forward(self, lm: torch.Tensor) -> torch.Tensor:
        """lm: (B,68,2)  →  (B,68,C_gcn)"""
        x = lm
        for layer in self.layers:
            x = layer(x, self.adj)
        return x


# ── AULandmarkPool ────────────────────────────────────────────────────────────

class AULandmarkPool(nn.Module):

    def __init__(self, pool_indices=None):
        super().__init__()
        self.pool_indices = pool_indices if pool_indices is not None else _AU_POOL_IDX

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [node_feats[:, idx, :].mean(dim=1) for idx in self.pool_indices],
            dim=1
        )   # (B, N_a, C_gcn)


# ── AdaptiveFusion ────────────────────────────────────────────────────────────

class AdaptiveFusion(nn.Module):

    def __init__(self, gcn_channels: int, emb_dim: int, num_aus: int = 8):
        super().__init__()
        self.proj    = nn.Linear(gcn_channels, emb_dim)
        self.bn_proj = nn.BatchNorm1d(num_aus)
        # sigmoid(-2.2) ≈ 0.1  →  bắt đầu với ảnh hưởng nhỏ từ geometry
        self.alpha_logit = nn.Parameter(torch.full((num_aus,), -2.2))

    def forward(self, V_a: torch.Tensor, V_geom: torch.Tensor) -> torch.Tensor:
        V_proj  = self.bn_proj(self.proj(V_geom))                   # (B,N_a,EMB_DIM)
        alpha   = torch.sigmoid(self.alpha_logit).unsqueeze(0).unsqueeze(-1)  # (1,N_a,1)
        return V_a + alpha * V_proj                                 # (B,N_a,EMB_DIM)

    def get_alpha(self) -> torch.Tensor:
        """Trả về gate α của từng AU, dùng để phân tích/visualize."""
        return torch.sigmoid(self.alpha_logit).detach().cpu()
