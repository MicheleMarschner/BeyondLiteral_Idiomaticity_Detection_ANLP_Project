from pathlib import Path
import torch
import os
from dataclasses import dataclass
import nltk


def is_colab() -> bool:
    return "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ


def is_kaggle() -> bool:
    if os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return True
    return False


@dataclass(frozen=True)
class Paths:
    """Holds the main filesystem paths used by the project"""
    data_raw: Path
    data_preprocessed: Path
    data_analysis: Path
    results: Path
    runs: Path
    checkpoints: Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root (…/BeyondLiteral_Idiomaticity_Detection)
IN_COLAB = is_colab()
IN_KAGGLE = (not IN_COLAB) and is_kaggle()
IN_DOCKER = os.getenv("IN_DOCKER", "0") == "1"


if IN_DOCKER:
    PATHS = Paths(
        data_raw=Path("/data/raw"),
        data_preprocessed=Path("/data/preprocessed"),
        data_analysis=Path("/data/analysis"),
        results=Path("/results"),
        runs=Path("/experiments"),
        checkpoints=Path("/checkpoints")
    )
elif IN_KAGGLE: 
    kaggle_root = Path("/kaggle/working/code/idiomaticity-code").resolve()
    if kaggle_root.exists():                 
        PROJECT_ROOT = kaggle_root

    PATHS = Paths(
        data_raw=PROJECT_ROOT / "data" / "raw",
        data_preprocessed=PROJECT_ROOT / "data" / "preprocessed",
        data_analysis=PROJECT_ROOT / "data" / "analysis",
        results=Path("/kaggle/working/results"),
        runs=Path("/kaggle/working/experiments"),
        checkpoints=Path("/kaggle/working/checkpoints")
    )
else:
    PATHS = Paths(
        data_raw=PROJECT_ROOT / "data" / "raw",
        data_preprocessed=PROJECT_ROOT / "data" / "preprocessed",
        data_analysis=PROJECT_ROOT / "data" / "analysis",
        results=PROJECT_ROOT / "results",
        runs=PROJECT_ROOT / "experiments",
        checkpoints=PROJECT_ROOT / "checkpoints"
    )


NLTK_DATA_DIR = PROJECT_ROOT / ".nltk_data"
nltk.data.path.insert(0, str(NLTK_DATA_DIR))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Data configs
MIN_TRAIN = 500
MIN_DEV = 100
MIN_TEST = 200

# require at least k samples per class in each split
MIN_PER_CLASS_TRAIN = 50
MIN_PER_CLASS_DEV = 20
MIN_PER_CLASS_TEST = 20