import os
import numpy as np
import pydicom
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

HU_MIN         = -1000
HU_MAX         =  400
TARGET_SPACING = (1.5, 1.5, 1.5)  # mm isotropic


def load_dicom_volume(patient_dir: str):
    slices = []
    for f in sorted(Path(patient_dir).glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(f))
            slices.append(ds)
        except Exception:
            continue

    if not slices:
        raise ValueError(f"No DICOM files found in {patient_dir}")

    # Sort by z-position
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    volume = np.stack([
        s.pixel_array.astype(np.float32) * float(s.RescaleSlope)
        + float(s.RescaleIntercept)
        for s in slices
    ])

    spacing = (
        float(slices[0].SliceThickness),
        float(slices[0].PixelSpacing[0]),
        float(slices[0].PixelSpacing[1]),
    )
    return volume, spacing


def resample_volume(volume: np.ndarray, original_spacing: tuple) -> np.ndarray:
    sitk_vol = sitk.GetImageFromArray(volume)
    sitk_vol.SetSpacing(list(reversed(original_spacing)))

    orig_size    = sitk_vol.GetSize()
    orig_spacing = sitk_vol.GetSpacing()
    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / TARGET_SPACING[-(i+1)]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(list(reversed(TARGET_SPACING)))
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(sitk_vol.GetDirection())
    resampler.SetOutputOrigin(sitk_vol.GetOrigin())
    resampler.SetDefaultPixelValue(HU_MIN)

    return sitk.GetArrayFromImage(resampler.Execute(sitk_vol)).astype(np.float32)


def window_normalize(volume: np.ndarray) -> np.ndarray:
    volume = np.clip(volume, HU_MIN, HU_MAX)
    return (volume - HU_MIN) / (HU_MAX - HU_MIN)


def preprocess_patient(patient_dir: str, out_path: str) -> None:
    volume, spacing = load_dicom_volume(patient_dir)
    volume = resample_volume(volume, spacing)
    volume = window_normalize(volume)
    np.save(out_path, volume)
    print(f"Saved: {out_path}  shape={volume.shape}")


def preprocess_all(dicom_root: str, output_root: str) -> None:
    os.makedirs(output_root, exist_ok=True)
    patient_dirs = sorted(Path(dicom_root).iterdir())

    for p in tqdm(patient_dirs, desc="Preprocessing"):
        if not p.is_dir():
            continue
        out_path = os.path.join(output_root, f"{p.name}.npy")
        if os.path.exists(out_path):
            continue  # skip already processed
        try:
            preprocess_patient(str(p), out_path)
        except Exception as e:
            print(f"  SKIP {p.name}: {e}")


if __name__ == "__main__":
    from src.utils.config import DICOM_DIR, PROCESSED_DIR
    preprocess_all(str(DICOM_DIR / "train"), str(PROCESSED_DIR / "train"))
    preprocess_all(str(DICOM_DIR / "test"),  str(PROCESSED_DIR / "test"))