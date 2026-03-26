import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
from src.training.loss import laplace_metric


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Sex_enc"]     = (df["Sex"] == "Male").astype(float)
    df["Smoking_enc"] = df["SmokingStatus"].map({
        "Never smoked":     0.0,
        "Ex-smoker":        1.0,
        "Currently smokes": 2.0
    }).fillna(0.0)
    df["WeekOffset"]    = df.groupby("Patient")["Weeks"].transform(lambda x: x - x.min())
    df["BaselineFVC"]   = df.groupby("Patient")["FVC"].transform("first")
    df["WeeksFromBase"] = df["Weeks"]
    return df


# Patient-level features (one row per patient)
PATIENT_FEATURES = ["Age", "Sex_enc", "Smoking_enc", "BaselineFVC"]

# Row-level features (vary per measurement)
ROW_FEATURES = ["WeeksFromBase", "WeekOffset", "BaselineFVC",
                "Age", "Sex_enc", "Smoking_enc"]

LGB_PARAMS = dict(
    objective        = "quantile",
    n_estimators     = 300,
    learning_rate    = 0.05,
    num_leaves       = 15,        # small — only ~140 training patients
    min_child_samples = 5,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = 42,
    verbose          = -1,
)


def train_baseline(train_csv: str, n_splits: int = 5) -> dict:
    df = pd.read_csv(train_csv)
    df = prepare_features(df)

    print("Feature sample:")
    print(df[ROW_FEATURES + ["FVC"]].head(3).to_string())
    print(f"\nFVC range : {df['FVC'].min():.0f} – {df['FVC'].max():.0f} ml")
    print(f"Patients  : {df['Patient'].nunique()}\n")

    X      = df[ROW_FEATURES].values.astype(np.float64)
    y      = df["FVC"].values.astype(np.float64)

    # Patient-level split — critical: no patient leaks across folds
    patients   = df["Patient"].unique()
    patient_kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_mu    = np.zeros(len(df))
    oof_sigma = np.zeros(len(df))

    for fold, (tr_pat_idx, val_pat_idx) in enumerate(patient_kf.split(patients)):
        tr_patients  = set(patients[tr_pat_idx])
        val_patients = set(patients[val_pat_idx])

        tr_mask  = df["Patient"].isin(tr_patients)
        val_mask = df["Patient"].isin(val_patients)

        X_tr, y_tr   = X[tr_mask],  y[tr_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        print(f"Fold {fold+1} | train patients: {len(tr_patients)} "
              f"({tr_mask.sum()} rows) | "
              f"val patients: {len(val_patients)} ({val_mask.sum()} rows)")

        m50 = lgb.LGBMRegressor(**{**LGB_PARAMS, "alpha": 0.50})
        m25 = lgb.LGBMRegressor(**{**LGB_PARAMS, "alpha": 0.25})
        m75 = lgb.LGBMRegressor(**{**LGB_PARAMS, "alpha": 0.75})

        m50.fit(X_tr, y_tr, feature_name=ROW_FEATURES)
        m25.fit(X_tr, y_tr, feature_name=ROW_FEATURES)
        m75.fit(X_tr, y_tr, feature_name=ROW_FEATURES)

        p50 = m50.predict(X_val)
        p25 = m25.predict(X_val)
        p75 = m75.predict(X_val)

        if fold == 0:
            print(f"  Q25 range : {p25.min():.0f} – {p25.max():.0f}")
            print(f"  Q75 range : {p75.min():.0f} – {p75.max():.0f}")
            print(f"  IQR mean  : {(p75 - p25).mean():.1f}")

        oof_mu[val_mask]    = p50
        oof_sigma[val_mask] = np.clip((p75 - p25) / 1.35, a_min=70, a_max=None)

        fold_score = laplace_metric(p50, oof_sigma[val_mask], y_val)
        print(f"  score: {fold_score:.4f} | "
              f"mu: {p50.min():.0f}–{p50.max():.0f} | "
              f"sigma mean: {oof_sigma[val_mask].mean():.1f}\n")

    overall = laplace_metric(oof_mu, oof_sigma, y)
    print(f"Overall OOF Laplace score : {overall:.4f}")
    print("(Anything better than -6.90 is a solid tabular baseline)")

    return {"oof_mu": oof_mu, "oof_sigma": oof_sigma, "score": overall}


if __name__ == "__main__":
    train_baseline("data/raw/train.csv")