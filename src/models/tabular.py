import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from src.training.loss import laplace_metric


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    le_sex     = LabelEncoder()
    le_smoking = LabelEncoder()

    df["Sex"]           = le_sex.fit_transform(df["Sex"])
    df["SmokingStatus"] = le_smoking.fit_transform(df["SmokingStatus"])
    df["FVCpct_norm"]   = df["Percent"] / 100.0

    # Week offset from each patient's earliest measurement
    df["WeekOffset"] = df.groupby("Patient")["Weeks"].transform(
        lambda x: x - x.min()
    )
    return df


FEATURES = ["Weeks", "WeekOffset", "Age", "Sex",
            "SmokingStatus", "FVC", "FVCpct_norm"]


def train_baseline(train_csv: str, n_splits: int = 10) -> dict:
    df   = pd.read_csv(train_csv)
    df   = prepare_features(df)
    X, y = df[FEATURES].values, df["FVC"].values

    kf        = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_mu    = np.zeros(len(df))
    oof_sigma = np.zeros(len(df))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        q50 = QuantileRegressor(quantile=0.50, alpha=0.01, solver="highs")
        q25 = QuantileRegressor(quantile=0.25, alpha=0.01, solver="highs")
        q75 = QuantileRegressor(quantile=0.75, alpha=0.01, solver="highs")

        q50.fit(X_tr, y_tr)
        q25.fit(X_tr, y_tr)
        q75.fit(X_tr, y_tr)

        oof_mu[val_idx]    = q50.predict(X_val)
        oof_sigma[val_idx] = np.clip(
            (q75.predict(X_val) - q25.predict(X_val)) / 1.35,
            a_min=70, a_max=None
        )

        fold_score = laplace_metric(
            oof_mu[val_idx], oof_sigma[val_idx], y_val
        )
        print(f"Fold {fold+1:02d} | score: {fold_score:.4f}")

    overall = laplace_metric(oof_mu, oof_sigma, y)
    print(f"\nOverall OOF Laplace score: {overall:.4f}")
    print("(Target: better than -6.90 to justify adding CT stream)")

    return {
        "oof_mu":    oof_mu,
        "oof_sigma": oof_sigma,
        "score":     overall
    }


if __name__ == "__main__":
    train_baseline("data/raw/train.csv")