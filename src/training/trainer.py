import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from pathlib import Path

from src.models.vit2d import SliceViT
from src.data.dataset import OSICDataset
from src.models.tabular import prepare_features
from src.training.loss import laplace_metric


def laplace_nll(mu, sigma, y):
    delta = (y - mu).abs().clamp(max=1000.0)
    return ((delta * 1.41421 / sigma) + torch.log(sigma * 1.41421)).mean()


def train_phase2(
    train_csv     : str,
    processed_dir : str,
    n_splits      : int   = 5,
    epochs        : int   = 30,
    lr            : float = 2e-4,
    num_slices    : int   = 15,
    batch_size    : int   = 4,
):
    df     = pd.read_csv(train_csv)
    df     = prepare_features(df)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ── Filter to only patients with preprocessed volumes ──────────
    processed_path = Path(processed_dir)
    available_ids  = {p.stem for p in processed_path.glob("*.npy")}
    before         = df["Patient"].nunique()
    df             = df[df["Patient"].isin(available_ids)].reset_index(drop=True)
    after          = df["Patient"].nunique()
    print(f"Device: {device} | Patients: {after}/{before} have CT volumes | Rows: {len(df)}")
    # ───────────────────────────────────────────────────────────────

    patients = df["Patient"].unique()
    kf       = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_mu    = np.zeros(len(df))
    oof_sigma = np.zeros(len(df))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(patients)):
        tr_pats  = set(patients[tr_idx])
        val_pats = set(patients[val_idx])

        tr_df  = df[df["Patient"].isin(tr_pats)].reset_index(drop=True)
        val_df = df[df["Patient"].isin(val_pats)].reset_index(drop=True)
        val_mask = df["Patient"].isin(val_pats)

        print(f"Fold {fold+1} | train: {len(tr_pats)} pts | val: {len(val_pats)} pts")

        tr_ds  = OSICDataset(tr_df,  processed_dir, num_slices, augment=True)
        val_ds = OSICDataset(val_df, processed_dir, num_slices, augment=False)

        tr_dl  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

        model = SliceViT(num_slices=num_slices).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=1e-6
        )

        scaler = torch.cuda.amp.GradScaler()  # mixed precision

        best_score, best_mu, best_sigma = -999, None, None

        for epoch in range(epochs):
            # ── Train ──────────────────────────────────────────────
            model.train()
            train_loss = 0
            for slices, tab, fvc in tr_dl:
                slices = slices.to(device)
                tab    = tab.to(device)
                fvc    = fvc.to(device)

                with torch.cuda.amp.autocast():
                    mu, sigma = model(slices, tab)
                    loss      = laplace_nll(mu, sigma, fvc)

                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                train_loss += loss.item()

            sched.step()

            # ── Validate ───────────────────────────────────────────
            model.eval()
            all_mu, all_sigma, all_y = [], [], []
            with torch.no_grad():
                for slices, tab, fvc in val_dl:
                    slices = slices.to(device)
                    tab    = tab.to(device)
                    with torch.cuda.amp.autocast():
                        mu, sigma = model(slices, tab)
                    all_mu.append(mu.cpu())
                    all_sigma.append(sigma.cpu())
                    all_y.append(fvc)

            all_mu    = torch.cat(all_mu).numpy()
            all_sigma = torch.cat(all_sigma).numpy()
            all_y     = torch.cat(all_y).numpy()
            score     = laplace_metric(all_mu, all_sigma, all_y)

            avg_loss  = train_loss / len(tr_dl)
            print(f"  epoch {epoch+1:2d}/{epochs} | "
                  f"train_loss: {avg_loss:.4f} | "
                  f"val_score: {score:.4f} | "
                  f"sigma: {all_sigma.mean():.1f}")

            if score > best_score:
                best_score = score
                best_mu    = all_mu.copy()
                best_sigma = all_sigma.copy()
                torch.save(model.state_dict(),
                           f"outputs/vit2d_fold{fold+1}_best.pth")

        oof_mu[val_mask]    = best_mu
        oof_sigma[val_mask] = best_sigma
        print(f"  Fold {fold+1} best: {best_score:.4f}\n")

    overall = laplace_metric(oof_mu, oof_sigma, df["FVC"].values)
    print(f"Overall OOF score (2D ViT) : {overall:.4f}")
    print(f"Baseline (tabular only)    : -7.39")
    print(f"Delta                      : {overall - (-7.39):+.4f}")

    return {"oof_mu": oof_mu, "oof_sigma": oof_sigma, "score": overall}


if __name__ == "__main__":
    train_phase2(
        train_csv     = "data/raw/train.csv",
        processed_dir = "data/processed/train",
    )