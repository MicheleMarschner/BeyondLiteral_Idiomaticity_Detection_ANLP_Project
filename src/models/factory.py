from typing import Any, Dict

from models.BERTs.BERTRunner import BERTRunner
from models.logreg.model import LogRegRunner

RUNNERS: Dict[str, Any] = {
    "logreg_tfidf": LogRegRunner,
    "logreg_word2vec": LogRegRunner,
    "mBERT": BERTRunner,
    "modernBERT": BERTRunner
}

def get_model_runner(model_family: str):
    """Return a fresh ModelRunner instance for the given model_family"""
    key = str(model_family).strip()

    runner = RUNNERS.get(key)
    if runner is None:
        raise ValueError(f"Unknown model_family='{key}'")

    return runner()
