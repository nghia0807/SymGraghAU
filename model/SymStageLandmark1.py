import torch
import torch.nn as nn

from .swin_transformer import (
    swin_transformer_tiny,
    swin_transformer_small,
    swin_transformer_base,
)
from .resnet import resnet18, resnet50, resnet101
from .basic_block import LinearBlock
from .landmark_gcn import LandmarkGCN, AULandmarkPool, AdaptiveFusion


EMB_DIM = 256   # đồng nhất với SymStage1


# ── Conv1DExtractor ───────────────────────────────────────────────────────────
# Copy y hệt từ SymStage1 để SymStageLandmark1 self-contained

class Conv1DExtractor(nn.Module):
    """(B, D, C_in) → (B, EMB_DIM) — giống hệt SymStage1."""

    def __init__(self, in_channels: int, hid_channels: int, emb_channels: int = EMB_DIM):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hid_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(hid_channels)
        self.conv2 = nn.Conv1d(hid_channels, emb_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(emb_channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):   # x: (B, D, C_in)
        x = x.transpose(1, 2)                            # (B, C_in, D)
        x = self.relu(self.bn1(self.conv1(x)))           # (B, C_hid, D)
        x = self.relu(self.bn2(self.conv2(x)))           # (B, EMB_DIM, D)
        return x.mean(dim=-1)                            # (B, EMB_DIM)


# ── ExprHead ──────────────────────────────────────────────────────────────────
# Copy y hệt từ SymStage1, KHÔNG thay đổi

class ExprHead(nn.Module):
    """Expression Head – giống hệt SymStage1."""

    def __init__(self, in_channels: int, num_expr: int, hid_channels: int):
        super().__init__()
        self.extractors = nn.ModuleList([
            Conv1DExtractor(in_channels, hid_channels) for _ in range(num_expr)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(EMB_DIM, 1) for _ in range(num_expr)
        ])

    def forward(self, x):   # x: (B, D, C_mid)
        embs, logits = [], []
        for ext, clf in zip(self.extractors, self.classifiers):
            e = ext(x)
            embs.append(e.unsqueeze(1))
            logits.append(clf(e))
        V_e = torch.cat(embs,   dim=1)    # (B, N_e, EMB_DIM)
        p_e = torch.sigmoid(torch.cat(logits, dim=1))  # (B, N_e)
        return V_e, p_e


# ── SymStageLandmark1 ─────────────────────────────────────────────────────────

class SymStageLandmark1(nn.Module):

    def __init__(self,
                 num_aus:      int   = 8,
                 num_expr:     int   = 7,
                 backbone:     str   = 'resnet50',
                 gcn_hidden:   tuple = (64, 128),
                 use_landmark: bool  = True):
        super().__init__()
        self.num_aus      = num_aus
        self.use_landmark = use_landmark

        # ── Backbone ────────────────────────────────────────────────────────
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()
            self.in_channels  = self.backbone.num_features
            self.mid_channels = self.in_channels // 2
            self.backbone.head = None
        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18()
            elif backbone == 'resnet101':
                self.backbone = resnet101()
            else:
                self.backbone = resnet50()
            self.in_channels  = self.backbone.fc.weight.shape[1]
            self.mid_channels = self.in_channels // 4
            self.backbone.fc  = None
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # ── Feature projection (giống SymStage1) ────────────────────────────
        self.global_linear = LinearBlock(self.in_channels, self.mid_channels)

        # ── AU visual extractors (giống AUHead.extractors trong SymStage1) ──
        self.au_extractors = nn.ModuleList([
            Conv1DExtractor(self.mid_channels, self.mid_channels)
            for _ in range(num_aus)
        ])

        # ── Geometric branch ─────────────────────────────────────────────────
        gcn_out_ch = gcn_hidden[-1] if use_landmark else EMB_DIM
        if use_landmark:
            self.landmark_gcn = LandmarkGCN(in_channels=2, gcn_hidden=gcn_hidden)
            self.au_pool      = AULandmarkPool()
            self.fusion       = AdaptiveFusion(gcn_channels=gcn_out_ch,
                                               emb_dim=EMB_DIM,
                                               num_aus=num_aus)

        # ── AU classifiers (áp dụng SAU fusion) ─────────────────────────────
        self.au_classifiers = nn.ModuleList([
            nn.Linear(EMB_DIM, 1) for _ in range(num_aus)
        ])

        # ── Expression Head (KHÔNG ĐỔI so với SymStage1) ────────────────────
        self.expr_head = ExprHead(
            in_channels=self.mid_channels,
            num_expr=num_expr,
            hid_channels=self.mid_channels,
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, images: torch.Tensor,
                landmarks: torch.Tensor = None):
       
        B = images.size(0)

        # 1. Backbone + projection
        feat = self.global_linear(self.backbone(images))   # (B, D, C_mid)

        # 2. Visual AU features
        embs = [ext(feat).unsqueeze(1) for ext in self.au_extractors]
        V_a  = torch.cat(embs, dim=1)   # (B, N_a, EMB_DIM)

        # 3. Geometric AU features + Adaptive Fusion
        if self.use_landmark and landmarks is not None:
            gcn_out = self.landmark_gcn(landmarks)   # (B, 68, C_gcn)
            V_geom  = self.au_pool(gcn_out)          # (B, N_a, C_gcn)
            V_a     = self.fusion(V_a, V_geom)       # (B, N_a, EMB_DIM)  ← V_fused

        # 4. AU classification (dùng V_a sau fusion)
        logits = torch.cat(
            [clf(V_a[:, i, :]) for i, clf in enumerate(self.au_classifiers)],
            dim=1
        )   # (B, N_a)
        p_a = torch.sigmoid(logits)

        # 5. Expression head (không đổi)
        V_e, p_e = self.expr_head(feat)

        return V_a, V_e, p_a, p_e

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_alpha(self):
        
        if self.use_landmark:
            return self.fusion.get_alpha()
        return None
