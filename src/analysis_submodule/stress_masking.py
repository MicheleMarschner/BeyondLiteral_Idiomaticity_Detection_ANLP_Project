from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from config import PATHS
from utils.helper import read_json, write_json
from models.factory import get_model_runner
from data import load_data_splits, build_inputs_for_splits
from evaluation.metrics import compute_metrics, make_predictions
from training import get_model


def mask_first_occurrence(s: str, mwe: str, mask_token: str = "[MASK]") -> str:
    # replaces first occurrence anywhere
    return s.replace(mwe, mask_token, 1)

def mask_last_occurrence(s: str, mwe: str, mask_token: str = "[MASK]") -> str:
    # replaces last occurrence anywhere
    i = s.rfind(mwe)
    if i == -1:
        return s
    return s[:i] + mask_token + s[i + len(mwe):]

def mask_all_occurrences(s: str, mwe: str, mask_token: str = "[MASK]") -> str:
    # replaces all occurrences
    return s.replace(mwe, mask_token)

def mask_all_occurrences_n_mask(s: str, mwe: str, mask_token: str = "[MASK]") -> str:
    n = max(1, len(str(mwe).split()))
    masked = " ".join([mask_token] * n)   
    return s.replace(mwe, masked)


def apply_mask(
    df: pd.DataFrame,
    variant: str,
    text_col: str = "input",
    mwe_col: str = "MWE",
    mask_token: str = "[MASK]",
) -> pd.DataFrame:
    out = df.copy()
    texts = out[text_col].astype(str).tolist()
    mwes = out[mwe_col].astype(str).tolist()

    if variant == "first":
        new_texts = [mask_first_occurrence(s, mwe, mask_token) for s, mwe in zip(texts, mwes)]
    elif variant == "last":
        new_texts = [mask_last_occurrence(s, mwe, mask_token) for s, mwe in zip(texts, mwes)]
    elif variant == "both":
        new_texts = [mask_all_occurrences(s, mwe, mask_token) for s, mwe in zip(texts, mwes)]
    elif variant == "both_n_mask":
        new_texts = [mask_all_occurrences_n_mask(s, mwe, mask_token) for s, mwe in zip(texts, mwes)]
    else:
        raise ValueError(f"Unknown variant={variant}")

    out[text_col] = new_texts
    return out


def stress_test_one_run(exp_dir: Path) -> Optional[Dict[str, Any]]:
    exp_config_path = exp_dir / "experiment_config.json"
    if not exp_config_path.exists():
        return None

    experiment_config = read_json(exp_config_path)
    metrics_normal = read_json(exp_dir / "metrics.json")

    runner = get_model_runner(experiment_config["model_family"])

    # rebuild splits
    train_df, val_df, test_df = load_data_splits(experiment_config, PATHS.data_preprocessed)
    train_data, val_data, test_data = build_inputs_for_splits(train_df, val_df, test_df, experiment_config)
  
    # load saved model
    model, best_params = get_model(experiment_config, exp_dir, train_data, val_data, runner)

    # masked inference
    rows = []
    for variant in ["first", "last", "both", "both_n_mask"]:
        masked_test = apply_mask(test_data, variant=variant, text_col="input", mwe_col="MWE")

        _, test_loader, _ = runner.prepare_features(
            params=best_params,
            config=experiment_config,
            train_df=train_data,
            test_df=masked_test,
        )

        proba = runner.predict_proba(model, test_loader)
        preds = make_predictions(proba)
        metrics_masked = compute_metrics(masked_test["label"], preds)

        rows.append({
            "variant": variant,
            "macro_f1_masked": metrics_masked["macro_f1"],
            "delta_macro_f1": float(metrics_masked["macro_f1"] - metrics_normal["macro_f1"]),
            "macro_f1_normal": metrics_normal["macro_f1"],
        })

    res = {
        "run_dir": exp_dir.name,
        "setting": experiment_config.get("setting"),
        "language_mode": experiment_config.get("language_mode"),
        "language": experiment_config.get("language"),
        "model_family": experiment_config.get("model_family"),
        "seed": experiment_config.get("seed"),
        "normal": metrics_normal,
        "masked_variants": rows,
    }
    return res


def run_stress_masking_over_all_runs(experiments_root: Path, results_root: Path) -> None:
    all_rows = []

    for experiment_dir in sorted(experiments_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        if not (experiment_dir / "metrics.json").exists():
            continue

        res = stress_test_one_run(experiment_dir)
        if res is None:
            continue


        # save per-run artifact
        write_json(experiment_dir / "stress_masking.json", res)

        base = {
            "run_dir": res["run_dir"],
            "setting": res["setting"],
            "language_mode": res["language_mode"],
            "language": res["language"],
            "model_family": res["model_family"],
            "seed": res["seed"],
            "macro_f1_normal": res["normal"]["macro_f1"],
        }

        flat = base.copy()
        for r in res["masked_variants"]:
            v = r["variant"]
            flat[f"macro_f1_{v}"] = r["macro_f1_masked"]
            flat[f"delta_macro_f1_{v}"] = r["delta_macro_f1"]

        all_rows.append(flat)
    
    df = pd.DataFrame(all_rows)
    
    df["delta_macro_f1_both_n_minus_both"] = (
        df["delta_macro_f1_both_n_mask"] - df["delta_macro_f1_both"]
    )

    df = df.sort_values("delta_macro_f1_both")
    
    save_path = results_root / "stress_masking_summary.csv"
    df.to_csv(save_path, index=False)