# src/data/dataset.py
"""
PyTorch Dataset for OSIC multi-modal data (CT slices + tabular features).
Used in Phase 2 (SliceViT) and Phase 3 (3D Swin).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from sklearn.preprocessing import StandardScaler


# ── Constants ─────────────────────────────────────────────────────────
NUM_SLICES = 15
IMG_SIZE   = 224
TAB_FEATURES = ["WeekDelta", "BaselineFVC", "Age", "Sex_enc", "Smoking_enc"]


# ── Image transforms ──────────────────────────────────────────────────
TRAIN_TRANSFORM = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.Normalize(mean=[0.5], std=[0.5]),
])

VAL_TRANSFORM = T.Compose([
    T.Normalize(mean=[0.5], std=[0.5]),
])


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering — identical across Phase 1, 2, 3.
    No 'Percent' feature: 1st place Kaggle winner found it hurts performance.

    Features created:
        Sex_enc      : 1.0 if Male else 0.0
        Smoking_enc  : Never=0, Ex=1, Current=2
        BaselineFVC  : first FVC measurement per patient
        BaselineWeek : week of first measurement
        WeekDelta    : weeks since baseline CT
    """
    df = df.copy()
    df["Sex_enc"] = (df["Sex"] == "Male").astype(float)
    df["Smoking_enc"] = df["SmokingStatus"].map({
        "Never smoked":     0.0,
        "Ex-smoker":        1.0,
        "Currently smokes": 2.0,
    }).fillna(0.0)
    df["BaselineFVC"]  = df.groupby("Patient")["FVC"].transform("first")
    df["BaselineWeek"] = df.groupby("Patient")["Weeks"].transform("first")
    df["WeekDelta"]    = df["Weeks"] - df["BaselineWeek"]
    return df


class OSICDataset(Dataset):
    """
    Multi-modal dataset: CT slices + tabular features.

    Args:
        df            : DataFrame with engineered features (output of prepare_features)
        processed_dir : path to folder containing patient_id.npy volumes
        tab_features  : list of tabular feature column names
        num_slices    : number of uniformly sampled CT slices per scan
        img_size      : resize each slice to (img_size, img_size)
        transform     : torchvision transform applied to each slice
        scaler        : fitted StandardScaler; if None, fits on this df
        is_train      : unused flag (kept for API compatibility)

    Returns per __getitem__:
        slices  : (num_slices, img_size, img_size)  float32 in [-1,1] after transform
        tabular : (len(tab_features),)               float32, z-scored
        fvc     : scalar float32                     ground truth FVC (ml)
    """

    def __init__(
        self,
        df:            pd.DataFrame,
        processed_dir: str,
        tab_features:  list = TAB_FEATURES,
        num_slices:    int  = NUM_SLICES,
        img_size:      int  = IMG_SIZE,
        transform            = None,
        scaler               = None,
        is_train:      bool  = True,
    ):
        self.df            = df.reset_index(drop=True)
        self.processed_dir = Path(processed_dir)
        self.tab_features  = tab_features
        self.num_slices    = num_slices
        self.img_size      = img_size
        self.transform     = transform
        self.is_train      = is_train

        # Fit or apply StandardScaler on tabular features
        X = df[tab_features].values.astype(np.float32)
        if scaler is None:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            self.X = self.scaler.transform(X)

    def _load_slices(self, patient_id: str) -> torch.Tensor:
        """Load CT volume, sample num_slices uniformly, resize to img_size."""
        npy_path = self.processed_dir / f"{patient_id}.npy"
        vol      = np.load(str(npy_path))          # (D, H, W) in [0, 1]
        D        = vol.shape[0]
        idxs     = np.linspace(0, D - 1, self.num_slices).astype(int)
        slices   = []
        for i in idxs:
            img = Image.fromarray((vol[i] * 255).astype(np.uint8))
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            slices.append(np.array(img, dtype=np.float32) / 255.0)
        return torch.tensor(np.stack(slices), dtype=torch.float32)  # (N, H, W)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row    = self.df.iloc[i]
        slices = self._load_slices(row["Patient"])           # (N, H, W)
        tab    = torch.tensor(self.X[i], dtype=torch.float32)
        fvc    = torch.tensor(row["FVC"], dtype=torch.float32)

        # Apply transform slice-by-slice
        if self.transform:
            transformed = []
            for s in slices:
                s_pil = T.ToPILImage()(s.unsqueeze(0))
                s_t   = T.ToTensor()(s_pil)
                s_t   = self.transform(s_t).squeeze(0)
                transformed.append(s_t)
            slices = torch.stack(transformed)

        return slices, tab, fvc