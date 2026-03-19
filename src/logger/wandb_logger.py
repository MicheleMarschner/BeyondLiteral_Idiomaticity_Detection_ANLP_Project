import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from transformers import TrainerCallback

from config import WANDB_ENABLED

load_dotenv()



class WandbDevCurveCallback(TrainerCallback):
    """Log train/dev curves to W&B during Trainer runs"""

    def __init__(self, wandb_run=None):
        self.wandb_run = wandb_run

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.wandb_run is None or logs is None:
            return

        if "loss" in logs:
            self.wandb_run.log(
                {"train_loss": float(logs["loss"])},
                step=int(state.global_step),
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.wandb_run is None or metrics is None:
            return

        payload = {}

        if "eval_loss" in metrics:
            payload["dev_loss"] = float(metrics["eval_loss"])

        if "eval_macro-F1" in metrics:
            payload["dev_macro_f1"] = float(metrics["eval_macro-F1"])

        if payload:
            self.wandb_run.log(payload, step=int(state.global_step))


def is_wandb_enabled() -> bool:
    """Return whether W&B logging is enabled globally"""
    return bool(WANDB_ENABLED)


def init_wandb_run(config: dict[str, Any], run_dir: Path):
    """Initialize a W&B run if enabled, otherwise return None"""
    if not is_wandb_enabled():
        return None

    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    if not WANDB_API_KEY:
        print("[wandb] WANDB_API_KEY not set. Continuing without W&B.")
        return None

    try:
        import wandb
    except ImportError:
        print("[wandb] wandb is not installed. Continuing without W&B.")
        return None

    WANDB_ENTITY = os.getenv("WANDB_ENTITY")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT")

    project = WANDB_PROJECT
    entity = WANDB_ENTITY

    input_variant = config.get("input_variant", {})
    features = input_variant.get("features", [])
    features_str = "-".join(features) if features else "none"

    group = (
        f"{config.get('model_family')}"
        f"__{config.get('setting')}"
        f"__{config.get('language')}"
        f"__{config.get('input_variant', {}).get('context')}"
    )

    run_name = (
        f"{config.get('setting')}"
        f"__{config.get('language')}"
        f"__{input_variant.get('context')}"
        f"_{input_variant.get('include_mwe_segment')}"
        f"_{input_variant.get('transform')}"
        f"_{features_str}"
        f"__{config.get('model_family')}"
        f"__seed{config.get('seed')}"
    )

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            job_type=config.get("model_family"),
            name=run_name,
            config={
                "setting": config.get("setting"),
                "language_mode": config.get("language_mode"),
                "language": config.get("language"),
                "model_family": config.get("model_family"),
                "seed": config.get("seed"),
                "context": input_variant.get("context"),
                "features": input_variant.get("features"),
                "include_mwe_segment": input_variant.get("include_mwe_segment"),
                "transform": input_variant.get("transform"),
            },
            dir=str(run_dir),
        )
        return run

    except Exception as e:
        print(f"[wandb] Initialization failed ({e}). Continuing without W&B.")
        return None


def update_wandb_split_stats_summary(
    run,
    split_stats: dict | None,
    is_too_small: bool,
    reasons: list[str] | None,
) -> None:
    """Write split statistics and split-check results to W&B summary"""
    if run is None or split_stats is None:
        return

    n_stats = split_stats.get("n", {})
    label_counts = split_stats.get("label_counts", {})

    label_name_map = {
        0: "idiomatic",
        1: "literal",
    }

    for split in ["train", "dev", "test"]:
        if split in n_stats:
            run.summary[f"{split}_size"] = int(n_stats[split])

        split_counts = label_counts.get(split, {})
        for label, count in split_counts.items():
            label_name = label_name_map.get(label, f"label_{label}")
            run.summary[f"{split}_{label_name}_count"] = int(count)

    run.summary["split_is_too_small"] = bool(is_too_small)
    run.summary["split_check_reasons"] = reasons or []


def update_wandb_best_params(run, best_params: dict | None) -> None:
    """Update W&B config with the selected best hyperparameters"""
    if run is None or best_params is None:
        return

    run.config.update(
        {f"{k}": v for k, v in best_params.items()},
        allow_val_change=True,
    )


def update_wandb_best_curves_summary(run, best_curves: dict | None) -> None:
    """Update W&B summary with best-step from saved curves"""
    if run is None or best_curves is None:
        return

    if "best_step" in best_curves:
        run.summary["best_step"] = int(best_curves["best_step"])


def update_wandb_best_result_summary(run, best_result: dict | None) -> None:
    """Update W&B summary with best scalar tuning results"""
    if run is None or best_result is None:
        return

    if "best_dev_macro_f1" in best_result:
        run.summary["best_dev_macro_f1"] = float(best_result["best_dev_macro_f1"])

    if "best_train_macro_f1" in best_result:
        run.summary["best_train_macro_f1"] = float(best_result["best_train_macro_f1"])


def log_wandb_tuning_results_table(run, tuning_results: list[dict] | None) -> None:
    """Log tuning results as a W&B table"""
    if run is None or not tuning_results:
        return

    try:
        import pandas as pd
        import wandb
    except ImportError:
        return

    tuning_df = pd.DataFrame(tuning_results)
    run.log({"tuning_results": wandb.Table(dataframe=tuning_df)})


def log_wandb_final_metrics(run, metrics: dict) -> None:
    """Log final evaluation metrics to W&B summary"""
    if run is None:
        return

    if "overall" in metrics:
        overall = metrics["overall"]
        per_language = metrics.get("per_language", {})
    else:
        overall = metrics
        per_language = {}

    run.summary["test_accuracy"] = float(overall["accuracy"])
    run.summary["test_macro_precision"] = float(overall["macro_precision"])
    run.summary["test_macro_recall"] = float(overall["macro_recall"])
    run.summary["test_macro_f1"] = float(overall["macro_f1"])

    confusion_name_map = {
        "tp": "test_literal_tp",
        "fp": "test_literal_fp",
        "tn": "test_literal_tn",
        "fn": "test_literal_fn",
    }

    cm = overall.get("confusion_matrix_values", {})
    for key, value in cm.items():
        if key in confusion_name_map:
            run.summary[confusion_name_map[key]] = int(value)

    for lang, lang_metrics in per_language.items():
        run.summary[f"test_{lang}_accuracy"] = float(lang_metrics["accuracy"])
        run.summary[f"test_{lang}_macro_precision"] = float(lang_metrics["macro_precision"])
        run.summary[f"test_{lang}_macro_recall"] = float(lang_metrics["macro_recall"])
        run.summary[f"test_{lang}_macro_f1"] = float(lang_metrics["macro_f1"])


def log_wandb_artifacts(run, run_dir: Path) -> None:
    """Upload selected saved output files as a W&B artifact"""
    if run is None:
        return
    
    try:
        import wandb
    except ImportError:
        return

    updated_name = (
        f"{run.name}-outputs"
        .replace(",", "-")
        .replace(" ", "_")
        .replace("/", "-")
    )
    artifact = wandb.Artifact(name=updated_name, type="run_outputs")

    for filename in [
        "metrics.json",
        "metrics.csv",
        "split_stats.csv",
        "test_predictions.csv",
        "best_params.json",
        "learning_curves.json",
        "tuning_results.csv",
    ]:
        file_path = run_dir / filename
        if file_path.exists():
            artifact.add_file(str(file_path), name=filename)

    run.log_artifact(artifact)


def finish_wandb_run(run) -> None:
    """Finish a W&B run if it exists"""
    if run is not None:
        run.finish()