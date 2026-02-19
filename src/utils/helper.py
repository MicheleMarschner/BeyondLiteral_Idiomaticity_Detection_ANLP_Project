import os
import random
import shutil
import numpy as np
import torch
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Sequence, Any

from config import Paths


def set_seeds(seed: int=51, deterministic: bool=True) -> None:
    """Sets seeds for complete reproducibility across all libraries and operations"""

    # Python hashing (affects iteration order in some cases)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

        # CUDA deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if deterministic:
            # cuDNN determinism
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # CUDA matmul determinism (PyTorch recommends setting this env var)
            # Only needed for some CUDA versions/ops; harmless otherwise.
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    if deterministic:
        # Force deterministic algorithms when available
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(str(e))


    print(f"All random seeds set to {seed} for reproducibility")


def ensure_dir(p: Path) -> None:
    """Create directory and parent directories if it doesn't already exist"""
    p.mkdir(parents=True, exist_ok=True)


def ensure_dirs(paths: Paths) -> None:
    """Create all directories that should exist"""
    for p in [paths.data_raw, paths.data_preprocessed, paths.results, paths.runs]:
        ensure_dir(p)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Save a Python dict as a readable JSON file"""
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_json(path: Path) -> Any:
    """Load a JSON file and return it as Python objects"""
    with open(path, "r") as f:
            results = json.load(f)
    
    return results


def read_csv_data(
    csv_path: str | Path,
    *,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Read a CSV into a DataFrame"""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding=encoding)

    return df


def build_input_str(input_variant: Dict[str, Any]) -> str:
    """Build a compact identifier string for an input-variant from the experiment config"""
    input_type = input_variant['context']                 
    include_mwe_segment = input_variant['include_mwe_segment']
    transform = input_variant['transform']
    features = input_variant['features']

    s = {f.strip().lower() for f in features if f and f.strip()}
    features_str = ""

    if not s:
        features_str = "empty"
    elif s == {"ner"}:
        features_str = "ner"
    elif s == {"glosses"}:
        features_str = "glosses"
    elif s == {"ner", "glosses"}:
        features_str = "ner+glosses"
    else : 
        raise ValueError(f"Unsupported features combination: {features}")

    return f"{input_type}_{include_mwe_segment}_{transform}_{features_str}"


def build_experiment_identifier(experiment_config: Dict[str, Any]) -> str:
    """Create a unique, human-readable run ID from the experiment configuration"""
    setting = experiment_config['setting']
    language = experiment_config['language']
    model_family = experiment_config['model_family']
    seed = experiment_config['seed']
    input_variant = experiment_config['input_variant']  # this is a dict

    input_str = build_input_str(input_variant)

    return f"{setting}__{language}__{input_str}__{model_family}__seed{seed}"
    

def create_experiment_dir(experiment_config: Dict[str, Any], run_dir: Path, overwrite: bool=False) -> Path:
    """Create a run folder for an experiment to store outputs and artifacts; if it exists, fail unless `overwrite=True` (then recreate it)."""
    folder_name = build_experiment_identifier(experiment_config)
    experiment_dir = Path(run_dir) / folder_name

    if experiment_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run folder already exists: {experiment_dir}\n"
                f"Set overwrite=True to rerun and overwrite artifacts."
            )
        # overwrite=True -> wipe the old run folder for a clean rerun
        shutil.rmtree(experiment_dir)

    ensure_dir(experiment_dir)
    return experiment_dir


def to_numpy_int(y: Sequence[Any]) -> np.ndarray:
    """Convert list to int numpy array"""
    arr = np.asarray(list(y))
    return arr.astype(int)


def to_numpy_float(y: Sequence[Any]) -> np.ndarray:
    """Convert list to float numpy array"""
    arr = np.asarray(list(y))
    return arr.astype(float)


def get_ids_by_pair(
    df: pd.DataFrame,
    types_df: pd.DataFrame,
    col1: str,
    col2: str,
    id_col: str = "ID",
) -> Sequence:
    """
    Same behavior, implemented via merge.
    """
    if id_col not in df.columns:
        raise ValueError(f"{id_col=} not in df")
    for c in (col1, col2):
        if c not in df.columns:
            raise ValueError(f"{c=} not in df")
        if c not in types_df.columns:
            raise ValueError(f"{c=} not in types_df")

    keys = types_df[[col1, col2]].drop_duplicates()
    matched = df.merge(keys, on=[col1, col2], how="inner")
    return matched[id_col].tolist()


def copy_file(src_file: Path, dst_file: Path, overwrite: bool = False) -> Path:
    """Copy a dataset file to an analysis location"""

    if not src_file.exists():
        raise FileNotFoundError(f"Source file not found: {src_file}")

    ensure_dir(dst_file)

    if dst_file.exists():
        raise 

    shutil.copy2(src_file, dst_file)  # copy2 keeps timestamps/metadata
    return dst_file