from typing import Sequence

def macro_f1(
    gold: Sequence[int], 
    pred: Sequence[int],
) -> float:
    """
    Compute macro-averaged F1

    Inputs:
      gold: true class ids (len N)
      pred: predicted class ids (len N)

    Output:
      macro F1 as float in [0, 1]
    """
    pass

def accuracy(
    gold: Sequence[int], 
    pred: Sequence[int],
) -> float:
    """
    Inputs:
      gold: true class ids (len N)
      y_pred: predicted class ids (len N)

    Output:
      accuracy as float in [0, 1]
    """
    pass

def per_class_f1(
    gold: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    """
    Inputs:
      y_true, y_pred: length-N sequences of class ids
      labels: explicit class id order to evaluate (recommended for stability)
      label_names: optional names aligned with `labels` (same length)

    Output:
      dict mapping class name -> F1 (e.g., {"literal": 0.91, "idiomatic": 0.74})
    """