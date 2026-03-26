import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from src.training.loss import laplace_metric


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Sex_enc"]     = (df["Sex"] == "Male").astype(float)
    df["Smoking_enc"] = df["SmokingStatus"].map({
        "Never smoked":      0.0,
        "Ex-smoker":         1.0,
        "Currently smokes":  2.0
    }).fillna(0.0)

    df["WeekOffset"]  = df.groupby("Patient")["Weeks"].transform(
        lambda x: x - x.min()
    )
    df["BaselineFVC"] = df.groupby("Patient")["FVC"].transform("first")
    df["WeeksFromBase"] = df["Weeks"]

    return df


FEATURES = [
    "WeeksFromBase",
    "WeekOffset",
    "Age",
    "Sex_enc",
    "Smoking_enc",
    "BaselineFVC",
    "FVC"
]

LGB_PARAMS_BASE = dict(
    objective      = "quantile",
    n_estimators   = 500,
    learning_rate  = 0.05,
    num_leaves     = 31,
    min_child_samples = 10,
    subsample      = 0.8,
    colsample_bytree = 0.8,
    random_state   = 42,
    verbose        = -1,
)


def train_baseline(train_csv: str, n_splits: int = 10) -> dict:
    df = pd.read_csv(train_csv)
    df = prepare_features(df)

    print("Feature sample:")
    print(df[FEATURES].head(3).to_string())
    print(f"\nFVC range : {df['FVC'].min():.0f} – {df['FVC'].max():.0f} ml")
    print(f"Patients  : {df['Patient'].nunique()}\n")

    X      = df[FEATURES].values.astype(np.float64)
    y      = df["FVC"].values.astype(np.float64)
    groups = df["Patient"].values

    gkf       = GroupKFold(n_splits=n_splits)
    oof_mu    = np.zeros(len(df))
    oof_sigma = np.zeros(len(df))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr        = y[tr_idx]

        m50 = lgb.LGBMRegressor(**{**LGB_PARAMS_BASE, "alpha": 0.50})
        m25 = lgb.LGBMRegressor(**{**LGB_PARAMS_BASE, "alpha": 0.25})
        m75 = lgb.LGBMRegressor(**{**LGB_PARAMS_BASE, "alpha": 0.75})

        m50.fit(X_tr, y_tr)
        m25.fit(X_tr, y_tr)
        m75.fit(X_tr, y_tr)

        p50 = m50.predict(X_val)
        p25 = m25.predict(X_val)
        p75 = m75.predict(X_val)

        if fold == 0:
            print(f"  Q25 range : {p25.min():.0f} – {p25.max():.0f}")
            print(f"  Q75 range : {p75.min():.0f} – {p75.max():.0f}")
            print(f"  IQR mean  : {(p75 - p25).mean():.1f}")

        oof_mu[val_idx]    = p50
        oof_sigma[val_idx] = np.clip((p75 - p25) / 1.35, a_min=70, a_max=None)

        fold_score = laplace_metric(
            oof_mu[val_idx], oof_sigma[val_idx], y[val_idx]
        )
        print(f"Fold {fold+1:02d} | score: {fold_score:.4f} | "
              f"mu range: {p50.min():.0f}–{p50.max():.0f} | "
              f"sigma mean: {oof_sigma[val_idx].mean():.1f}")

    overall = laplace_metric(oof_mu, oof_sigma, y)
    print(f"\nOverall OOF Laplace score : {overall:.4f}")
    print("(Target to beat in Phase 2: this score + CT stream)")

    return {"oof_mu": oof_mu, "oof_sigma": oof_sigma, "score": overall}


if __name__ == "__main__":
    train_baseline("data/raw/train.csv")