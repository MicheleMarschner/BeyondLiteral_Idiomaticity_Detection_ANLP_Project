from typing import Sequence, Tuple, Dict, Any
import numpy as np

from utils.helper import to_numpy_int, to_numpy_float


def make_predictions(proba: Sequence[Any], threshold: float=0.5) -> np.ndarray:
    """Convert 1D literal-class probabilities into 0/1 predictions using a threshold."""
    
    proba = to_numpy_float(proba)
    assert proba.ndim == 1, (f"probabilities must be 1D after squeeze, got shape {proba.shape}")
    
    preds = (proba >= float(threshold)).astype(int)
    return preds


def compute_confusion_matrix_counts(gold_labels: np.ndarray, preds: np.ndarray) -> Dict[str, int]:
    """Compute TP/FP/TN/FN for binary classification with positive class=1."""

    tp = int(np.sum((gold_labels == 1) & (preds == 1)))
    fp = int(np.sum((gold_labels == 0) & (preds == 1)))
    tn = int(np.sum((gold_labels == 0) & (preds == 0)))
    fn = int(np.sum((gold_labels == 1) & (preds == 0)))

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_accuracy(correct: int, total: int) -> float:
    """Compute classification accuracy as the fraction of correct predictions over all predictions (returns 0.0 if total is 0)."""
    return (correct / total) if total > 0 else 0.0


def compute_precision_recall_f1_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Precision/recall/F1 for a class given tp/fp/fn."""
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) != 0 else 0.0

    return precision, recall, f1


def compute_macro_metrics(tp: int, tn: int, fp: int, fn: int) ->  Tuple[float, float, float]:
    """Macro-average precision/recall/F1 for binary classification (mean of pos and neg class metrics)."""
    pos_precision, pos_recall, pos_f1 = compute_precision_recall_f1_from_counts(tp=tp, fp=fp, fn=fn)
    neg_precision, neg_recall, neg_f1 = compute_precision_recall_f1_from_counts(tp=tn, fp=fn, fn=fp)

    macro_precision = 0.5 * (pos_precision + neg_precision)
    macro_recall = 0.5 * (pos_recall + neg_recall)
    macro_f1 = 0.5 * (pos_f1 + neg_f1)

    return macro_precision, macro_recall, macro_f1


def compute_metrics(
    gold_labels: Sequence[int], 
    preds: Sequence[int],
) -> Dict[str, Any]:
    """Compute accuracy, macro P/R/F1, and confusion-matrix counts for binary predictions."""
    
    gold_labels = to_numpy_int(gold_labels)
    preds = to_numpy_int(preds)

    assert len(preds) == len(gold_labels), (f"Length mismatch between preds={len(preds)} and gold_labels={len(gold_labels)}")

    confusion_matrix_values = compute_confusion_matrix_counts(gold_labels, preds)
    tp, fp, tn, fn = confusion_matrix_values['tp'], confusion_matrix_values['fp'], confusion_matrix_values['tn'], confusion_matrix_values['fn']

    accuracy = compute_accuracy(tp + tn, tp + tn + fp + fn)
    macro_precision, macro_recall, macro_f1 = compute_macro_metrics(tp, tn, fp, fn)

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix_values": confusion_matrix_values, 
    }


def compute_metrics_per_language(
    gold_labels: Sequence[int],
    preds: Sequence[float],   
    languages: Sequence[str],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    
    gold_labels = to_numpy_int(gold_labels)
    preds = to_numpy_float(preds)
    langs = np.asarray(languages)

    out: Dict[str, Any] = {
        "overall": compute_metrics(gold_labels, preds),
        "per_language": {},
    }

    langs_unique = np.unique(langs)

    for lang in langs_unique:
        lang_mask = (langs == lang)              # boolean mask
        gold_lang = gold_labels[lang_mask]       
        preds_lang = preds[lang_mask]            

        out["per_language"][str(lang)] = compute_metrics(gold_lang, preds_lang)

    return out