from typing import Sequence, Any

## standardize per experiment run: id, lang, setting (zero/one-shot), input_variant (T0/T1), 
## ggf. mwe_type (normalized string), gold, pred, prob or confidence

## output: experiment settings, predictions, metrics, loss

def compute_eval_metrics(
    gold: Sequence[int], 
    pred: Sequence[int],
) -> dict[str, Any]:
    """Returns macro_f1, accuracy and per_class_f1"""
    pass