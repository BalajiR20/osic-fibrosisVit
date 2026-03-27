import torch
import torch.nn as nn
import timm
import numpy as np


class SliceViT(nn.Module):
    """
    2D ViT applied per CT slice.
    Takes N uniformly sampled slices per patient,
    encodes each with ViT, aggregates across slices,
    then fuses with tabular features to predict mu + sigma.
    """
    def __init__(
        self,
        vit_name    = "vit_small_patch16_224",
        num_slices  = 15,
        tab_dim     = 5,
        embed_dim   = 384,   # vit_small output dim
        dropout     = 0.3,
    ):
        super().__init__()
        self.num_slices = num_slices

        # ── Image backbone ──────────────────────────────────────────
        # Pretrained on ImageNet — we adapt it to single-channel CT
        self.vit = timm.create_model(
            vit_name,
            pretrained   = True,
            num_classes  = 0,       # remove classification head
            in_chans     = 1,       # CT is grayscale
        )

        # ── Slice aggregation ────────────────────────────────────────
        # Learned weighted average across slices
        self.slice_attn = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # ── Tabular encoder ──────────────────────────────────────────
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
        )

        # ── Fusion + prediction head ─────────────────────────────────
        fused_dim = embed_dim + 128
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.SiLU(),
        )
        self.mu_head    = nn.Linear(64, 1)
        self.sigma_head = nn.Linear(64, 1)

        # Init sigma to predict ~200ml initially
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.constant_(self.sigma_head.bias, 5.3)

    def encode_slices(self, slices):
        """
        slices: (B, N, 1, H, W)
        Returns: (B, embed_dim) — one vector per patient
        """
        B, N, C, H, W = slices.shape

        # Encode all slices in one batch pass
        slices_flat = slices.view(B * N, C, H, W)          # (B*N, 1, H, W)
        feats_flat  = self.vit(slices_flat)                 # (B*N, embed_dim)
        feats       = feats_flat.view(B, N, -1)             # (B, N, embed_dim)

        # Attention-weighted aggregation across slices
        attn_w = self.slice_attn(feats)                     # (B, N, 1)
        attn_w = torch.softmax(attn_w, dim=1)               # (B, N, 1)
        img_emb = (feats * attn_w).sum(dim=1)               # (B, embed_dim)
        return img_emb

    def forward(self, slices, tab):
        """
        slices : (B, N, 1, H, W)  — N CT slices per patient
        tab    : (B, tab_dim)     — tabular features
        Returns: mu (B,), sigma (B,)
        """
        img_emb = self.encode_slices(slices)         # (B, embed_dim)
        tab_emb = self.tab_encoder(tab)              # (B, 128)

        fused   = torch.cat([img_emb, tab_emb], dim=1)
        h       = self.head(fused)

        mu    = self.mu_head(h).squeeze(1) * 4000.0
        sigma = nn.functional.softplus(
            self.sigma_head(h).squeeze(1)
        ) + 70.0

        return mu, sigma