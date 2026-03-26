import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.training.loss import laplace_metric


# ─── Feature Engineering ────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Sex_enc"]     = (df["Sex"] == "Male").astype(float)
    df["Smoking_enc"] = df["SmokingStatus"].map({
        "Never smoked":     0.0,
        "Ex-smoker":        1.0,
        "Currently smokes": 2.0
    }).fillna(0.0)
    # Baseline = first FVC reading per patient
    df["BaselineFVC"]  = df.groupby("Patient")["FVC"].transform("first")
    df["BaselineWeek"] = df.groupby("Patient")["Weeks"].transform("first")
    df["WeekDelta"]    = df["Weeks"] - df["BaselineWeek"]
    return df


# ─── Dataset ─────────────────────────────────────────────────────────────────

class FVCDataset(Dataset):
    def __init__(self, df):
        self.fvc        = torch.tensor(df["FVC"].values,        dtype=torch.float32)
        self.week_delta = torch.tensor(df["WeekDelta"].values,  dtype=torch.float32)
        self.base_fvc   = torch.tensor(df["BaselineFVC"].values,dtype=torch.float32)
        self.age        = torch.tensor(df["Age"].values,        dtype=torch.float32)
        self.sex        = torch.tensor(df["Sex_enc"].values,    dtype=torch.float32)
        self.smoke      = torch.tensor(df["Smoking_enc"].values,dtype=torch.float32)

    def __len__(self):
        return len(self.fvc)

    def __getitem__(self, i):
        x = torch.stack([
            self.week_delta[i] / 100.0,   # normalize weeks
            self.base_fvc[i]   / 4000.0,  # normalize FVC
            self.age[i]        / 80.0,
            self.sex[i],
            self.smoke[i]      / 2.0,
        ])
        return x, self.fvc[i]


# ─── Model ───────────────────────────────────────────────────────────────────

class QuantileNet(nn.Module):
    """
    Simple MLP that outputs mu (median FVC) and log_sigma (uncertainty).
    Trained with Laplace log-likelihood directly.
    This is the architecture that dominated OSIC — simple but effective.
    """
    def __init__(self, in_dim=5, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 2)   # [mu, log_sigma]
        )
        # Initialize mu output near baseline FVC scale
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, x):
        out       = self.net(x)
        mu        = out[:, 0] * 4000.0   # scale back to FVC range
        log_sigma = out[:, 1]
        sigma     = torch.exp(log_sigma).clamp(min=70.0)
        return mu, sigma


# ─── Loss ─────────────────────────────────────────────────────────────────────

def laplace_loss(mu, sigma, fvc_true):
    delta = (fvc_true - mu).abs().clamp(max=1000.0)
    sigma = sigma.clamp(min=70.0)
    return (-(delta * 1.4142 / sigma) - torch.log(sigma * 1.4142)).mean()


# ─── Training ─────────────────────────────────────────────────────────────────

def train_one_fold(tr_df, val_df, epochs=600, lr=1e-3, device="cpu"):
    tr_ds  = FVCDataset(tr_df)
    val_ds = FVCDataset(val_df)
    tr_dl  = DataLoader(tr_ds,  batch_size=128, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = QuantileNet().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_score, best_mu, best_sigma = -999, None, None

    for epoch in range(epochs):
        model.train()
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            mu, sigma = model(x)
            loss = -laplace_loss(mu, sigma, y)   # minimise negative score
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        # Validate every 50 epochs
        if (epoch + 1) % 50 == 0:
            model.eval()
            all_mu, all_sigma, all_y = [], [], []
            with torch.no_grad():
                for x, y in val_dl:
                    x = x.to(device)
                    mu, sigma = model(x)
                    all_mu.append(mu.cpu()); all_sigma.append(sigma.cpu()); all_y.append(y)
            all_mu    = torch.cat(all_mu).numpy()
            all_sigma = torch.cat(all_sigma).numpy()
            all_y     = torch.cat(all_y).numpy()
            score = laplace_metric(all_mu, all_sigma, all_y)
            if score > best_score:
                best_score = score
                best_mu    = all_mu.copy()
                best_sigma = all_sigma.copy()

    return best_mu, best_sigma, best_score


# ─── Cross-Validation ─────────────────────────────────────────────────────────

def train_baseline(train_csv: str, n_splits: int = 5, epochs: int = 600) -> dict:
    df      = pd.read_csv(train_csv)
    df      = prepare_features(df)
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Patients: {df['Patient'].nunique()} | Rows: {len(df)}\n")

    patients   = df["Patient"].unique()
    kf         = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_mu    = np.zeros(len(df))
    oof_sigma = np.zeros(len(df))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(patients)):
        tr_pats  = set(patients[tr_idx])
        val_pats = set(patients[val_idx])

        tr_df  = df[df["Patient"].isin(tr_pats)].reset_index(drop=True)
        val_df = df[df["Patient"].isin(val_pats)].reset_index(drop=True)

        print(f"Fold {fold+1} | train: {len(tr_pats)} pts | val: {len(val_pats)} pts")

        val_mask = df["Patient"].isin(val_pats)
        mu, sigma, score = train_one_fold(tr_df, val_df, epochs=epochs, device=device)

        oof_mu[val_mask]    = mu
        oof_sigma[val_mask] = sigma

        print(f"  score: {score:.4f} | mu: {mu.min():.0f}–{mu.max():.0f} | "
              f"sigma mean: {sigma.mean():.1f}\n")

    overall = laplace_metric(oof_mu, oof_sigma, df["FVC"].values)
    print(f"Overall OOF Laplace score : {overall:.4f}")
    print("(Kaggle 1st place tabular-only was around -6.75)")

    return {"oof_mu": oof_mu, "oof_sigma": oof_sigma, "score": overall}


if __name__ == "__main__":
    train_baseline("data/raw/train.csv")