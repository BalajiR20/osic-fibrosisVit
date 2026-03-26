import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold
from src.training.loss import laplace_metric


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Encode categoricals with fixed mapping (not fit_transform — that can vary per fold)
    df["Sex_enc"]    = (df["Sex"] == "Male").astype(float)
    df["Smoking_enc"] = df["SmokingStatus"].map({
        "Never smoked": 0.0,
        "Ex-smoker":    1.0,
        "Currently smokes": 2.0
    }).fillna(0.0)

    # Week offset from each patient's first measurement
    df["WeekOffset"] = df.groupby("Patient")["Weeks"].transform(
        lambda x: x - x.min()
    )

    # Baseline FVC (first measurement per patient)
    baseline = df.groupby("Patient")["FVC"].transform("first")
    df["BaselineFVC"] = baseline

    # Weeks since baseline CT (negative weeks exist in this dataset)
    df["WeeksFromBase"] = df["Weeks"]

    return df


# No Percent feature — 1st place winner found it hurts
FEATURES = [
    "WeeksFromBase",
    "WeekOffset",
    "Age",
    "Sex_enc",
    "Smoking_enc",
    "BaselineFVC",
    "FVC"
]


def train_baseline(train_csv: str, n_splits: int = 10) -> dict:
    df = pd.read_csv(train_csv)
    df = prepare_features(df)

    # Debug: check features look sensible
    print("Feature sample:")
    print(df[FEATURES].head(3).to_string())
    print(f"\nFVC range: {df['FVC'].min():.0f} – {df['FVC'].max():.0f} ml")
    print(f"Weeks range: {df['Weeks'].min()} – {df['Weeks'].max()}")
    print(f"Patients: {df['Patient'].nunique()}\n")

    X   = df[FEATURES].values.astype(np.float64)
    y   = df["FVC"].values.astype(np.float64)
    groups = df["Patient"].values

    # Scale features — critical for QuantileRegressor convergence
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # GroupKFold — ensures no patient appears in both train and val
    # This is correct for this dataset (patient-level split)
    gkf = GroupKFold(n_splits=n_splits)

    oof_mu    = np.zeros(len(df))
    oof_sigma = np.zeros(len(df))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr        = y[tr_idx]

        q50 = QuantileRegressor(quantile=0.50, alpha=0.001, solver="highs")
        q25 = QuantileRegressor(quantile=0.25, alpha=0.001, solver="highs")
        q75 = QuantileRegressor(quantile=0.75, alpha=0.001, solver="highs")

        q50.fit(X_tr, y_tr)
        q25.fit(X_tr, y_tr)
        q75.fit(X_tr, y_tr)

        oof_mu[val_idx] = q50.predict(X_val)

        iqr = q75.predict(X_val) - q25.predict(X_val)
        oof_sigma[val_idx] = np.clip(iqr / 1.35, a_min=70, a_max=None)

        fold_score = laplace_metric(
            oof_mu[val_idx], oof_sigma[val_idx], y[val_idx]
        )
        print(f"Fold {fold+1:02d} | score: {fold_score:.4f} | "
              f"mu range: {oof_mu[val_idx].min():.0f}–{oof_mu[val_idx].max():.0f} | "
              f"sigma mean: {oof_sigma[val_idx].mean():.1f}")

    overall = laplace_metric(oof_mu, oof_sigma, y)
    print(f"\nOverall OOF Laplace score : {overall:.4f}")
    print(f"Score spread across folds : {np.std([laplace_metric(oof_mu[val_idx], oof_sigma[val_idx], y[val_idx]) for _, val_idx in gkf.split(X, y, groups)]):.4f}")

    return {
        "oof_mu":    oof_mu,
        "oof_sigma": oof_sigma,
        "score":     overall,
        "scaler":    scaler
    }


if __name__ == "__main__":
    train_baseline("data/raw/train.csv")