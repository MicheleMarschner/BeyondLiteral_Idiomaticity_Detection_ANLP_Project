from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from config import PATHS
from utils.helper import read_json, write_json
from models.factory import get_model_runner
from data import load_data_splits, build_inputs_for_splits
from evaluation.metrics import compute_metrics, make_predictions
from training import get_model


def mask_mwe(
    df: pd.DataFrame,
    text_col: str = "input",
    mwe_col: str = "mwe",
    mask_token: str = "[MASK]",
    preserve_word_count: bool = True,
) -> pd.DataFrame:
    new_df = df.copy()

    new_text = []
    for s, mwe in zip(new_df[text_col].astype(str), new_df[mwe_col].astype(str)):
        if preserve_word_count:
            n = max(1, len(mwe.split()))
            masked = " ".join([mask_token] * n)
        else:
            masked = mask_token
        new_text.append(s.replace(mwe, masked, 1))

    new_df[text_col] = new_text
    return new_df


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

    print("[DEBUG] tokenizer_source =", best_params.get("tokenizer_source"))

    # masked inference
    masked_test = mask_mwe(test_data, mwe_col="MWE")
    _, test_loader, _ = runner.prepare_features(
        params=best_params,
        config=experiment_config,
        train_df=train_data,
        test_df=masked_test,
    )
    proba_masked = runner.predict_proba(model, test_loader)
    preds_masked= make_predictions(proba_masked)
    metrics_masked = compute_metrics(masked_test["label"], preds_masked)

    res = {
        "run_name": exp_dir.name,
        "setting": experiment_config.get("setting"),
        "language_mode": experiment_config.get("language_mode"),
        "language": experiment_config.get("language"),
        "model_family": experiment_config.get("model_family"),
        "seed": experiment_config.get("seed"),
        "normal": metrics_normal,
        "masked": metrics_masked,
        "delta_macro_f1": float(metrics_masked["macro_f1"] - metrics_normal["macro_f1"]),
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

        print(f"{experiment_dir.name}: Δmacro_f1={res['delta_macro_f1']:.4f}")

    df = pd.DataFrame(all_rows).sort_values("delta_macro_f1")
    
    save_path = results_root / "stress_masking_summary.csv"
    df.to_csv(save_path, index=False)