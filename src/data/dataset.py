import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path


def sample_slices(volume: np.ndarray, n: int = 15) -> np.ndarray:
    """
    Uniformly sample N slices from a 3D CT volume.
    volume: (D, H, W) float32 in [0, 1]
    Returns: (N, 1, 224, 224)
    """
    D = volume.shape[0]
    indices = np.linspace(0, D - 1, n).astype(int)
    slices  = volume[indices]                        # (N, H, W)

    # Resize each slice to 224x224
    from PIL import Image
    resized = []
    for s in slices:
        img = Image.fromarray((s * 255).astype(np.uint8))
        img = img.resize((224, 224), Image.BILINEAR)
        resized.append(np.array(img) / 255.0)

    slices_out = np.stack(resized)[:, np.newaxis, :, :]  # (N, 1, 224, 224)
    return slices_out.astype(np.float32)


def get_tab_features(row: pd.Series) -> np.ndarray:
    """
    Extract normalized tabular features for one row.
    """
    sex_enc    = 1.0 if row["Sex"] == "Male" else 0.0
    smoke_map  = {"Never smoked": 0.0, "Ex-smoker": 1.0, "Currently smokes": 2.0}
    smoke_enc  = smoke_map.get(row["SmokingStatus"], 0.0)

    return np.array([
        row["WeekDelta"]   / 100.0,
        row["BaselineFVC"] / 4000.0,
        row["Age"]         / 80.0,
        sex_enc,
        smoke_enc          / 2.0,
    ], dtype=np.float32)


class OSICDataset(Dataset):
    """
    Each item = one (patient, week) prediction target.
    Loads the preprocessed CT volume and samples N slices.
    """
    def __init__(
        self,
        df           : pd.DataFrame,
        processed_dir: str,
        num_slices   : int  = 15,
        augment      : bool = False,
    ):
        self.df            = df.reset_index(drop=True)
        self.processed_dir = Path(processed_dir)
        self.num_slices    = num_slices
        self.augment       = augment

        # Cache: patient_id -> volume (loaded once per patient)
        self._vol_cache = {}

    def _load_volume(self, patient_id: str) -> np.ndarray:
        if patient_id not in self._vol_cache:
            path = self.processed_dir / f"{patient_id}.npy"
            self._vol_cache[patient_id] = np.load(str(path))
        return self._vol_cache[patient_id]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        vol     = self._load_volume(row["Patient"])
        slices  = sample_slices(vol, n=self.num_slices)

        if self.augment:
            # Horizontal flip (left-right lung symmetry)
            if np.random.rand() > 0.5:
                slices = slices[:, :, :, ::-1].copy()
            # Intensity jitter
            slices = np.clip(
                slices + np.random.uniform(-0.05, 0.05), 0, 1
            ).astype(np.float32)

        tab = get_tab_features(row)
        fvc = np.float32(row["FVC"])

        return (
            torch.from_numpy(slices),   # (N, 1, 224, 224)
            torch.from_numpy(tab),      # (5,)
            torch.tensor(fvc),          # scalar
        )