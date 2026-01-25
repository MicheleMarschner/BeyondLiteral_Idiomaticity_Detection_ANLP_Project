from pathlib import Path
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

@dataclass(frozen=True)
class Paths:
    """Project-wide fixed container paths"""
    raw: Path
    processed: Path
    results: Path
    checkpoints: Path

# Fixed locations inside the Docker container
PATHS = Paths(
    raw=Path("/data/raw"),
    processed=Path("/data/processed"),
    results=Path("/results"),
    checkpoints=Path("/results/checkpoints"),
)


def ensure_dirs(paths: Paths=PATHS) -> None:
    """Create all directories that should exist"""
    for p in [paths.raw, paths.processed, paths.results, paths.checkpoints]:
        p.mkdir(parents=True, exist_ok=True)



# --- core enums ---
Lang = Literal["EN", "PT", "GL"]
SplitSetting = Literal["zero_shot", "one_shot"]
InputView = Literal["T0", "T1", "T2"]
Transform = Literal["none", "mask_mwe", "mwe_only", "context_only"]

ModelFamily = Literal[
    "tfidf_logreg_word",
    "static_embed_logreg",
    "transformer_cls_frozen",
]


@dataclass(frozen=True)
class DatasetSpec:
    """
    Defines which official training file/split setting is used and which languages are included.
    """
    setting: SplitSetting                 # zero_shot | one_shot
    train_langs: Tuple[Lang, ...]         # ("EN",) or ("EN","PT") etc.
    dev_langs: Tuple[Lang, ...] = ("EN", "PT", "GL")  # usually evaluate per-language or choose subset

    # Optional: if you ever want to freeze exact file names
    train_split_name: Optional[str] = None   # e.g. "train_zero" / "train_one"
    dev_split_name: str = "dev"              # dev_gold is labels; still call split "dev"


@dataclass(frozen=True)
class InputSpec:
    """
    Controls what text goes into the model.
    """
    view: InputView = "T0"                 # T0 target only; T1 prev+target+next; T2 marker view
    transform: Transform = "none"          # diagnostics: mask_mwe | mwe_only | context_only
    mask_token: str = "<MWE>"              # used if transform == mask_mwe

    # For marker/span models (T2/transformer_spanpool)
    marker_left: str = "<mwe>"
    marker_right: str = "</mwe>"


@dataclass(frozen=True)
class RunSpec:
    """
    Run-wide settings for reproducibility and bookkeeping.
    """
    seed: int = 51
    active: bool = True
    tags: Tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class ExperimentConfig:
    """
    One atomic experiment = (dataset choice) x (input choice) x (model family + params).
    """
    name: str
    family: ModelFamily

    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    input: InputSpec = field(default_factory=InputSpec)
    run: RunSpec = field(default_factory=RunSpec)

    # model-specific hyperparameters live here (vectorizer params, transformer params, etc.)
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializable config for saving alongside metrics."""
        return asdict(self)
