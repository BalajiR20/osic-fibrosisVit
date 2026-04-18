# src/training/trainer.py
"""
Training logic for Phase 2 (SliceViT).

Main function: train_phase2()
    - Loads data, runs 5-fold GroupKFold CV
    - Two-phase training: warmup (frozen ViT) then full (top-half ViT unfrozen)
    - Saves best checkpoint per fold to checkpoints/
    - Returns OOF predictions for metric computation
"""

import os
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

from src.data.dataset import (
    OSICDataset, prepare_features,
    TAB_FEATURES, TRAIN_TRANSFORM, VAL_TRANSFORM
)
from src.models.vit2d import SliceViT
from src.training.loss import LaplaceLoss, laplace_metric


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def train_one_fold(
    train_df:      pd.DataFrame,
    val_df:        pd.DataFrame,
    processed_dir: str,
    fold:          int,
    device:        str,
    epochs:        int   = 40,
    lr:            float = 2e-4,
    batch_size:    int   = 4,
    warmup_epochs: int   = 5,
    patience:      int   = 10,
    checkpoint_dir:str   = "checkpoints",
) -> tuple:
    """
    Train SliceViT for one CV fold.

    Two-phase training:
        Phase A — warmup_epochs: ViT fully frozen, only fusion + heads update
        Phase B — remaining epochs: top half of ViT unfrozen, 10× lower LR

    Returns:
        best_mu    : OOF predictions for val patients at best epoch
        best_sigma : OOF uncertainty predictions at best epoch
        val_fvc    : ground truth FVC for val patients
        best_score : best val Laplace score achieved
        history    : dict with train_loss and val_score per epoch
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Datasets & Loaders ────────────────────────────────────────────
    train_ds = OSICDataset(
        df=train_df, processed_dir=processed_dir,
        tab_features=TAB_FEATURES, transform=TRAIN_TRANSFORM, is_train=True
    )
    val_ds = OSICDataset(
        df=val_df, processed_dir=processed_dir,
        tab_features=TAB_FEATURES, transform=VAL_TRANSFORM,
        scaler=train_ds.scaler, is_train=False
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = SliceViT(tab_dim=len(TAB_FEATURES)).to(device)
    # ViT is frozen by default from __init__

    # ── Optimizer (warmup: only non-ViT params) ───────────────────────
    optimizer = torch.optim.AdamW(
        model.get_non_vit_params(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    loss_fn   = LaplaceLoss()
    scaler_amp = torch.cuda.amp.GradScaler()

    best_score   = -np.inf
    best_mu      = None
    best_sigma   = None
    patience_cnt = 0
    history      = {"train_loss": [], "val_score": []}

    for epoch in range(1, epochs + 1):

        # ── Switch from warmup to full training ───────────────────────
        if epoch == warmup_epochs + 1:
            vit_unfrozen = model.unfreeze_vit_top_half()
            optimizer = torch.optim.AdamW([
                {"params": vit_unfrozen,              "lr": lr * 0.1},
                {"params": model.get_non_vit_params(), "lr": lr},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 0.01
            )
            print(f"    [Fold {fold}] Epoch {epoch}: ViT top half unfrozen → full training")

        # ── Train ─────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for slices, tabular, fvc in train_loader:
            slices, tabular, fvc = (
                slices.to(device), tabular.to(device), fvc.to(device)
            )
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                mu, sigma = model(slices, tabular)
                loss = loss_fn(mu, sigma, fvc)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        val_mus, val_sigmas, val_fvcs = [], [], []
        with torch.no_grad():
            for slices, tabular, fvc in val_loader:
                slices, tabular = slices.to(device), tabular.to(device)
                with torch.cuda.amp.autocast():
                    mu, sigma = model(slices, tabular)
                val_mus.append(mu.cpu().numpy())
                val_sigmas.append(sigma.cpu().numpy())
                val_fvcs.append(fvc.numpy())

        val_mu    = np.concatenate(val_mus)
        val_sigma = np.concatenate(val_sigmas)
        val_fvc   = np.concatenate(val_fvcs)
        val_score = laplace_metric(val_mu, val_sigma, val_fvc)

        history["train_loss"].append(avg_loss)
        history["val_score"].append(val_score)

        # Log every 5 epochs
        if epoch % 5 == 0 or epoch == 1 or epoch == warmup_epochs + 1:
            print(
                f"    [Fold {fold}] epoch {epoch:03d}/{epochs} | "
                f"train_loss: {avg_loss:.4f} | "
                f"val_score: {val_score:.4f} | "
                f"sigma_mean: {val_sigma.mean():.1f} ml"
            )

        # Early stopping + best model checkpoint
        if val_score > best_score:
            best_score   = val_score
            best_mu      = val_mu.copy()
            best_sigma   = val_sigma.copy()
            patience_cnt = 0
            ckpt_path = os.path.join(checkpoint_dir, f"slicevit_fold{fold}.pt")
            torch.save({
                "model_state_dict":   model.state_dict(),
                "scaler":             train_ds.scaler,
                "fold":               fold,
                "best_score":         best_score,
                "tab_features":       TAB_FEATURES,
            }, ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"    [Fold {fold}] Early stop at epoch {epoch} (patience={patience})")
                break

    return best_mu, best_sigma, val_fvc, best_score, history


def train_phase2(
    train_csv:     str,
    processed_dir: str,
    n_splits:      int   = 5,
    epochs:        int   = 40,
    lr:            float = 2e-4,
    batch_size:    int   = 4,
    warmup_epochs: int   = 5,
    patience:      int   = 10,
    seed:          int   = 42,
    checkpoint_dir:str   = "checkpoints",
) -> dict:
    """
    Full 5-fold cross-validation training for Phase 2 SliceViT.

    Args:
        train_csv     : path to train.csv
        processed_dir : path to folder with patient .npy CT volumes
        n_splits      : number of CV folds
        epochs        : max epochs per fold
        lr            : base learning rate
        batch_size    : CT scans per batch (keep ≤ 4 for T4 GPU)
        warmup_epochs : epochs with ViT frozen
        patience      : early stopping patience (epochs)
        seed          : random seed for reproducibility
        checkpoint_dir: where to save fold checkpoints

    Returns:
        dict with keys:
            oof_mu, oof_sigma, oof_fvc : OOF predictions
            fold_scores                : list of best val score per fold
            overall_score              : OOF Laplace metric
            fold_histories             : list of training history dicts
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load and filter to patients with CT volumes ───────────────────
    df = pd.read_csv(train_csv)
    df = prepare_features(df)

    available_ids = {Path(f).stem for f in Path(processed_dir).glob("*.npy")}
    n_before = df["Patient"].nunique()
    df       = df[df["Patient"].isin(available_ids)].reset_index(drop=True)
    n_after  = df["Patient"].nunique()

    print(f"Device   : {device}")
    print(f"Patients : {n_after}/{n_before} have CT volumes | Rows: {len(df):,}")
    print(f"Folds    : {n_splits} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"Warmup   : {warmup_epochs} epochs (ViT frozen)")
    print("=" * 65)

    # ── GroupKFold — no patient leaks across folds ────────────────────
    gkf      = GroupKFold(n_splits=n_splits)
    patients = df["Patient"].values

    oof_mu        = np.zeros(len(df))
    oof_sigma     = np.zeros(len(df))
    oof_fvc       = np.zeros(len(df))
    fold_scores   = []
    fold_histories = []

    for fold, (tr_idx, val_idx) in enumerate(
        gkf.split(df, groups=patients), start=1
    ):
        print(f"\nFold {fold}/{n_splits} | "
              f"train: {len(set(patients[tr_idx]))} pts | "
              f"val: {len(set(patients[val_idx]))} pts")

        tr_df  = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        mu, sigma, fvc, best, history = train_one_fold(
            train_df=tr_df,
            val_df=val_df,
            processed_dir=processed_dir,
            fold=fold,
            device=device,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            warmup_epochs=warmup_epochs,
            patience=patience,
            checkpoint_dir=checkpoint_dir,
        )

        oof_mu[val_idx]    = mu
        oof_sigma[val_idx] = sigma
        oof_fvc[val_idx]   = fvc
        fold_scores.append(best)
        fold_histories.append(history)

        print(f"  Fold {fold} BEST  : {best:.4f} | "
              f"mu: {mu.min():.0f}–{mu.max():.0f} | "
              f"sigma_mean: {sigma.mean():.1f} ml")

    # ── OOF metric ────────────────────────────────────────────────────
    overall = laplace_metric(oof_mu, oof_sigma, oof_fvc)
    print("\n" + "=" * 65)
    for i, s in enumerate(fold_scores, 1):
        print(f"  Fold {i}: {s:.4f}")
    print("=" * 65)
    print(f"  Overall OOF Laplace score : {overall:.4f}")
    print(f"  Fold std                  : {np.std(fold_scores):.4f}")
    print()
    print(f"  Phase 1 baseline : -6.6716")
    print(f"  Phase 2 (this)   : {overall:.4f}   Δ = {overall - (-6.6716):+.4f}")
    print(f"  FVC-Net SOTA     : -6.6414   Gap = {overall - (-6.6414):+.4f}")

    return {
        "oof_mu":         oof_mu,
        "oof_sigma":      oof_sigma,
        "oof_fvc":        oof_fvc,
        "fold_scores":    fold_scores,
        "overall_score":  overall,
        "fold_histories": fold_histories,
        "df":             df,
    }