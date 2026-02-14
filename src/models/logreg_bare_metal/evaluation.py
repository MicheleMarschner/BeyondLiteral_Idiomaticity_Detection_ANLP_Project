import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd


def save_logreg_feature_weights(
    model,
    featurizer,
    out_path: str,
    label0_name: str = "idiomatic",
    label1_name: str = "literal",
) -> pd.DataFrame:
    """
    Save a CSV with one row per feature (term/ngram) and its logistic-regression weight.

    Convention here:
      - weight > 0 pushes toward class 1 (literal)
      - weight < 0 pushes toward class 0 (idiomatic)
    """
    feature_names = featurizer.get_feature_names_out()
    w = np.asarray(model.weights).ravel()

    if len(feature_names) != len(w):
        raise ValueError(f"Feature/weight mismatch: {len(feature_names)} features vs {len(w)} weights.")
    

    direction = np.where(w > 0, label1_name, np.where(w < 0, label0_name, "neutral"))

    df = pd.DataFrame(
        {
            "feature": feature_names.astype(str),
            "weight": w.astype(float),
            "abs_weight": np.abs(w).astype(float),
            "direction": direction,
        }
    ).sort_values("abs_weight", ascending=False)

    df.to_csv(out_path, index=False)
    return df


def top_features_by_class(path: str, k: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a saved feature-weight CSV and return the top-k features for each class:
      - idiomatic (class 0): most negative weights
      - literal  (class 1): most positive weights
    """
    df = pd.read_csv(path)

    if "weight" not in df.columns or "feature" not in df.columns:
        raise ValueError("CSV must contain at least 'feature' and 'weight' columns.")

    idiomatic_top = df.sort_values("weight", ascending=True).head(k)          # most negative
    literal_top   = df.sort_values("weight", ascending=False).head(k)         # most positive

    return idiomatic_top, literal_top
    

def explain_one(text: str, model, featurizer, top_k: int = 20):
    """
    Show which TF-IDF features pushed this prediction toward idiomatic (0) vs literal (1).
    Negative contribution -> idiomatic, positive -> literal.

    This gives you a ranked list like:

feature, tfidf, weight, contribution

So you can say: “this sentence was predicted idiomatic mainly because ‘spill the beans’ had TF-IDF=..., weight=..., contribution=...”
    """
    x = featurizer.transform([text])  # (1, V) CSR
    w = np.asarray(model.weights).ravel()
    b = float(model.bias)

    row = x.getrow(0)
    idxs, vals = row.indices, row.data
    names = featurizer.get_feature_names_out()

    contribs = []
    for j, tfidf_val in zip(idxs, vals):
        contrib = float(tfidf_val * w[j])
        contribs.append((names[j], float(tfidf_val), float(w[j]), contrib))

    contribs.sort(key=lambda t: abs(t[3]), reverse=True)

    # model output
    logit = float(row @ w + b)                 # log-odds for class 1 (literal)
    proba_literal = 1.0 / (1.0 + np.exp(-logit))
    pred = 1 if proba_literal >= 0.5 else 0

    return {
        "pred": pred,  # 0 idiomatic, 1 literal
        "proba_literal": float(proba_literal),
        "top_contribs": contribs[:top_k],
    }


"""
If both means are almost the same and std is tiny → features aren’t separating classes (tokenization/ngrams/min_df issue or learning rate too low / too much regularization).
debug signal
"""

def logits(model, X):
    w = np.asarray(model.weights).ravel()
    return np.asarray(X @ w).ravel() + float(model.bias)

def class_logit_stats(model, X, y):
    z = logits(model, X)
    y = np.asarray(y).ravel()
    return {
        "mean_logit_y0": float(z[y==0].mean()) if np.any(y==0) else None,
        "mean_logit_y1": float(z[y==1].mean()) if np.any(y==1) else None,
        "std_logit": float(z.std()),
    }

