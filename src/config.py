from pathlib import Path
import torch
import os
from dataclasses import dataclass

def is_colab() -> bool:
    return "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ


def is_kaggle() -> bool:
    if Path("/kaggle/input").exists():
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
IN_KAGGLE = is_kaggle()
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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Data configs
MIN_TRAIN = 500
MIN_DEV = 200
MIN_TEST = 200

# require at least k samples per class in each split
MIN_PER_CLASS_TRAIN = 100
MIN_PER_CLASS_DEV = 50
MIN_PER_CLASS_TEST = 50