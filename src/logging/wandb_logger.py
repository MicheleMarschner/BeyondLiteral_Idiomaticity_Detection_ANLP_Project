
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
import wandb
from wandb.sdk.wandb_run import Run
import json

from config import WANDB_ENABLED

load_dotenv()



def is_wandb_enabled() -> bool:
    """Return whether W&B logging is enabled globally"""
    return bool(WANDB_ENABLED)


def init_wandb_run(config: dict[str, Any], run_dir: Path):
    """Initialize a W&B run if enabled, otherwise return None."""
    if not is_wandb_enabled():
        return None

    input_variant = config.get("input_variant", {})
    features = input_variant.get("features", [])
    features_str = "-".join(features) if features else "none"

    group = (
        f"{config.get('setting')}"
        f"__{config.get('language_mode')}"
        f"__{config.get('language')}"
        f"__{input_variant.get('context')}"
        f"__{input_variant.get('transform')}"
        f"__mwe{input_variant.get('include_mwe_segment')}"
        f"__feat-{features_str}"
    )

    run_name = (
        f"{config.get('model_family')}"
        f"__{config.get('setting')}"
        f"__{config.get('language')}"
        f"__seed{config.get('seed')}"
    )

    run = wandb.init(
        project="anlp-idioms",
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


def log_wandb_final_metrics(run, metrics: dict[str, Any]) -> None:
    """Log final evaluation metrics to W&B"""
    if run is None:
        return

    if "overall" in metrics:
        overall = metrics["overall"]
        per_language = metrics.get("per_language", {})
    else:
        overall = metrics
        per_language = {}

    run.summary["test/accuracy"] = float(overall["accuracy"])
    run.summary["test/macro_precision"] = float(overall["macro_precision"])
    run.summary["test/macro_recall"] = float(overall["macro_recall"])
    run.summary["test/macro_f1"] = float(overall["macro_f1"])

    cm = overall.get("confusion_matrix_values", {})
    for key in ["tp", "fp", "tn", "fn"]:
        if key in cm:
            run.summary[f"test/{key}"] = int(cm[key])

    for lang, lang_metrics in per_language.items():
        run.summary[f"test/lang/{lang}/accuracy"] = float(lang_metrics["accuracy"])
        run.summary[f"test/lang/{lang}/macro_precision"] = float(lang_metrics["macro_precision"])
        run.summary[f"test/lang/{lang}/macro_recall"] = float(lang_metrics["macro_recall"])
        run.summary[f"test/lang/{lang}/macro_f1"] = float(lang_metrics["macro_f1"])

        lang_cm = lang_metrics.get("confusion_matrix_values", {})
        for key in ["tp", "fp", "tn", "fn"]:
            if key in lang_cm:
                run.summary[f"test/lang/{lang}/{key}"] = int(lang_cm[key])


def log_wandb_learning_curves(run, run_dir: Path) -> None:
    """Log saved learning curves from learning_curves.json to W&B."""
    if run is None:
        return

    curves_path = run_dir / "learning_curves.json"
    if not curves_path.exists():
        return

    with curves_path.open("r", encoding="utf-8") as f:
        curves = json.load(f)

    train_steps = curves.get("train_steps", [])
    train_loss = curves.get("train_loss", [])
    for step, loss in zip(train_steps, train_loss):
        run.log({
            "step": int(step),
            "train/loss": float(loss),
        })

    dev_steps = curves.get("dev_steps", [])
    dev_loss = curves.get("dev_loss", [])
    dev_macro_f1 = curves.get("dev_macro_f1", [])

    for i, step in enumerate(dev_steps):
        payload = {"step": int(step)}
        if i < len(dev_loss):
            payload["dev/loss"] = float(dev_loss[i])
        if i < len(dev_macro_f1):
            payload["dev/macro_f1"] = float(dev_macro_f1[i])
        run.log(payload)

    best_step = curves.get("best_step")
    if best_step is not None:
        run.summary["best_step"] = int(best_step)


def log_wandb_artifacts(run, run_dir: Path) -> None:
    """Upload selected saved output files as a W&B artifact"""
    if run is None:
        return

    artifact = wandb.Artifact(name=f"{run.name}-outputs", type="run_outputs")

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