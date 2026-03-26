import yaml
from pathlib import Path

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Central paths — change DATA_ROOT if your data lives elsewhere
DATA_ROOT      = Path("data")
RAW_DIR        = DATA_ROOT / "raw"
DICOM_DIR      = DATA_ROOT / "dicom"
PROCESSED_DIR  = DATA_ROOT / "processed"
OUTPUTS_DIR    = Path("outputs")