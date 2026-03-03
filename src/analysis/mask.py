from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

from src.config import PATHS
from src.utils.helper import read_json, write_json
from src.models.factory import get_model_runner
from src.data import load_data_splits, build_inputs_for_splits
from src.evaluation.metrics import compute_metrics, make_predictions
from src.training import get_model


def mask_mwe_by_string(
    df: pd.DataFrame,
    text_col: str = "input",
    mwe_col: str = "mwe",
    mask_token: str = "[MASK]",
    preserve_word_count: bool = True,
) -> pd.DataFrame:
    out = df.copy()

    def _mask_row(row):
        s = str(row[text_col])
        mwe = str(row[mwe_col])

        if preserve_word_count:
            n = max(1, len(mwe.split()))
            masked = " ".join([mask_token] * n)
        else:
            masked = "<TARGET>"
        return s.replace(mwe, masked, 1)

    out[text_col] = out.apply(_mask_row, axis=1)
    return out


def mask_fn(df: pd.DataFrame):

    cols = set(df.columns)

    for c in ["mwe", "MWE", "expression", "target_expression"]:
        if c in cols:
            return lambda d, col=c: mask_mwe_by_string(d, mwe_col=col)
    
    return None


def stress_test_one_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    cfg_path = run_dir / "experiment_config.json"
    if not cfg_path.exists():
        return None

    experiment_config = read_json(cfg_path)

    runner = get_model_runner(experiment_config["model_family"])

    # rebuild splits for this exact config
    train_df, val_df, test_df = load_data_splits(experiment_config, PATHS.data_preprocessed)
    train_data, val_data, test_data = build_inputs_for_splits(train_df, val_df, test_df, experiment_config)
        
    # load saved model
    model, best_params = get_model(experiment_config, run_dir, train_data, val_data, runner)

    # normal inference
    _, test_obj, _ = runner.prepare_features(
        params=best_params,
        config=experiment_config,
        train_df=train_data,
        test_df=test_data,
    )
    proba = runner.predict_proba(model, test_obj)
    preds = make_predictions(proba, threshold=0.5)
    m_normal = compute_metrics(test_data["label"], preds)

    # masked inference
    masked_test = mask_fn(test_data)
    _, test_obj_m, _ = runner.prepare_features(
        params=best_params,
        config=experiment_config,
        train_df=train_data,
        test_df=masked_test,
    )
    proba_m = runner.predict_proba(model, test_obj_m)
    preds_m = make_predictions(proba_m, threshold=0.5)
    m_masked = compute_metrics(masked_test["label"], preds_m)

    res = {
        "run_dir": run_dir.name,
        "setting": experiment_config.get("setting"),
        "language_mode": experiment_config.get("language_mode"),
        "language": experiment_config.get("language"),
        "model_family": experiment_config.get("model_family"),
        "seed": experiment_config.get("seed"),
        "normal": m_normal,
        "masked": m_masked,
        "delta_macro_f1": float(m_masked["macro_f1"] - m_normal["macro_f1"]),
    }
    return res


def run_stress_masking_over_all_runs(experiments_root: Path, results_root: Path) -> None:
    all_rows = []

    for run_dir in sorted(experiments_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "metrics.json").exists():
            continue

        res = stress_test_one_run(run_dir)
        if res is None:
            continue

        # save per-run artifact
        write_json(run_dir / "stress_masking.json", res)

        # collect flat row
        all_rows.append({
            "setting": res["setting"],
            "language_mode": res["language_mode"],
            "language": res["language"],
            "model_family": res["model_family"],
            "seed": res["seed"],
            "macro_f1_normal": res["normal"]["macro_f1"],
            "macro_f1_masked": res["masked"]["macro_f1"],
            "delta_macro_f1": res["delta_macro_f1"],
        })

        print(f"[ok] {run_dir.name}: Δmacro_f1={res['delta_macro_f1']:.4f}")

    df = pd.DataFrame(all_rows).sort_values("delta_macro_f1")
    save_path = results_root / "stress_masking_summary.csv"
    df.to_csv(save_path, index=False)
    print(f"[done] wrote: {save_path}")


if __name__ == "__main__":
    results_root = PATHS.results
    experiments_root = PATHS.runs
    run_stress_masking_over_all_runs(experiments_root, results_root)