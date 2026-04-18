# src/models/vit2d.py
"""
SliceViT — 2D Vision Transformer for CT slice-level feature extraction.

Architecture:
    CT Volume (D, H, W)
        │
        ▼  sample_slices()  →  15 slices (15, 224, 224)
        │
        ▼  grayscale → RGB  →  (15, 3, 224, 224)
        │
        ▼  ViT-Small/16 (shared weights, pretrained ImageNet)
        │  → 15 CLS token embeddings  (15, 384)
        │
        ▼  mean pool across slices  →  CT embedding  (384,)
        │
        ├──  tabular encoder MLP  →  tab embedding  (64,)
        │
        ▼  concat  →  (448,)
        │
        ▼  fusion MLP (448 → 256 → 128)
        │
        ├──  mu_head   →  FVC prediction  (ml)
        └──  sig_head  →  uncertainty σ   (ml, ≥ 70)

Training strategy:
    Warmup (epochs 1–5) : ViT fully frozen, only fusion + heads train
    Full   (epoch 6+)   : unfreeze last 6 ViT blocks, 10× lower LR for ViT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from src.data.dataset import TAB_FEATURES, NUM_SLICES, IMG_SIZE


class SliceViT(nn.Module):
    """
    2D per-slice Vision Transformer with tabular fusion.

    Args:
        tab_dim    : number of tabular features (default: 5)
        tab_hidden : tabular encoder hidden dim (default: 64)
        vit_name   : timm model name (default: vit_small_patch16_224)
        vit_dim    : CLS token dimension from chosen ViT (384 for Small)
        fusion_dim : fusion MLP hidden dim (default: 256)
        num_slices : CT slices per scan (default: 15)
        dropout    : dropout rate (default: 0.3)
    """

    def __init__(
        self,
        tab_dim:    int   = len(TAB_FEATURES),
        tab_hidden: int   = 64,
        vit_name:   str   = "vit_small_patch16_224",
        vit_dim:    int   = 384,
        fusion_dim: int   = 256,
        num_slices: int   = NUM_SLICES,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.num_slices = num_slices
        self.vit_dim    = vit_dim

        # ── ViT backbone (shared across all slices) ────────────────────
        self.vit = timm.create_model(
            vit_name,
            pretrained=True,
            num_classes=0,      # remove classification head → outputs CLS token
        )

        # Freeze patch embed + first half of blocks initially
        # (unfreeze during training via unfreeze_vit())
        self._freeze_vit_for_warmup()

        # ── Tabular encoder ───────────────────────────────────────────
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_dim, tab_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(tab_hidden, tab_hidden),
            nn.SiLU(),
        )

        # ── Fusion MLP ────────────────────────────────────────────────
        fused_dim = vit_dim + tab_hidden
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fusion_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.SiLU(),
        )

        # ── Prediction heads ──────────────────────────────────────────
        self.mu_head  = nn.Linear(fusion_dim // 2, 1)
        self.sig_head = nn.Linear(fusion_dim // 2, 1)

        # Initialise heads to sensible starting values
        nn.init.constant_(self.mu_head.bias,  2700.0)   # mean FVC ~2700ml
        nn.init.constant_(self.sig_head.bias,  200.0)   # initial sigma ~200ml

    # ── Freeze / unfreeze helpers ──────────────────────────────────────

    def _freeze_vit_for_warmup(self):
        """Freeze all ViT parameters for warmup phase."""
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze_vit_top_half(self):
        """
        Unfreeze the top half of ViT blocks for full training.
        Called after warmup epochs are complete.
        Returns the ViT parameters that were unfrozen (for optimizer).
        """
        total_blocks = len(self.vit.blocks)
        freeze_until = total_blocks // 2
        unfrozen = []
        for name, param in self.vit.named_parameters():
            if "blocks" in name:
                block_num = int(name.split(".")[1])
                if block_num >= freeze_until:
                    param.requires_grad = True
                    unfrozen.append(param)
            elif "norm" in name:   # final LayerNorm
                param.requires_grad = True
                unfrozen.append(param)
        return unfrozen

    def get_non_vit_params(self):
        """Return all non-ViT parameters (tab encoder, fusion, heads)."""
        return (
            list(self.tab_encoder.parameters())
            + list(self.fusion.parameters())
            + list(self.mu_head.parameters())
            + list(self.sig_head.parameters())
        )

    # ── Forward ───────────────────────────────────────────────────────

    def encode_ct(self, slices: torch.Tensor) -> torch.Tensor:
        """
        Process batch of CT slices through shared ViT.

        Args:
            slices: (B, N, H, W)  — batch, num_slices, height, width  (grayscale)
        Returns:
            ct_emb: (B, vit_dim) — mean-pooled CLS embeddings
        """
        B, N, H, W = slices.shape
        # Grayscale → RGB by repeating channel (ViT was pretrained on RGB)
        x = slices.unsqueeze(2).repeat(1, 1, 3, 1, 1)  # (B, N, 3, H, W)
        x = x.reshape(B * N, 3, H, W)                   # (B*N, 3, H, W)

        # Forward through ViT — outputs CLS token embedding
        cls = self.vit(x)                               # (B*N, 384)
        cls = cls.view(B, N, -1)                         # (B, N, 384)

        return cls.mean(dim=1)                           # (B, 384) — mean pool

    def forward(
        self,
        slices:  torch.Tensor,   # (B, N, H, W)
        tabular: torch.Tensor,   # (B, tab_dim)
    ):
        ct_emb  = self.encode_ct(slices)                 # (B, 384)
        tab_emb = self.tab_encoder(tabular)              # (B, 64)

        fused   = torch.cat([ct_emb, tab_emb], dim=-1)  # (B, 448)
        h       = self.fusion(fused)                     # (B, 128)

        mu      = self.mu_head(h).squeeze(-1)            # (B,)
        sigma   = F.softplus(self.sig_head(h)).squeeze(-1) + 70.0   # (B,) ≥ 70

        return mu, sigma

    def get_attention_maps(self, slices: torch.Tensor, device: str):
        """
        Extract CLS attention maps from last ViT block for interpretability.

        Args:
            slices: (1, N, H, W) — single patient, N slices
        Returns:
            attn_maps: (N, num_patches) — attention weight of CLS over patches
        """
        self.eval()
        attentions = []

        def hook_fn(module, input, output):
            attentions.append(output.detach().cpu())

        handle = self.vit.blocks[-1].attn.attn_drop.register_forward_hook(hook_fn)

        B, N, H, W = slices.shape
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                x = slices.unsqueeze(2).repeat(1, 1, 3, 1, 1)
                x = x.reshape(B * N, 3, H, W).to(device)
                _ = self.vit(x)

        handle.remove()

        if not attentions:
            return None

        attn      = attentions[0]                        # (B*N, heads, seq, seq)
        cls_attn  = attn[:, :, 0, 1:].mean(dim=1)      # (B*N, num_patches)
        cls_attn  = cls_attn.view(B, N, -1)             # (B, N, num_patches)
        return cls_attn[0]                               # (N, num_patches)