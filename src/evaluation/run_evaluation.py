from pathlib import Path
import pandas as pd 

from config import PATHS
from evaluation.plots import plot_delta_bar, plot_en_pt_scatter, plot_heatmap_context_signal, plot_one_shot_gain
from utils.helper import ensure_dir, read_json


def flatten_run(run_dir: Path) -> list[dict]:
    exp_config = read_json(run_dir / "experiment_config.json")
    metrics = read_json(run_dir / "metrics.json")

    input_variant = exp_config["input_variant"]
    base = {
        "run_dir": run_dir.name,
        "setting": exp_config["setting"],
        "language_mode": exp_config["language_mode"],
        "language": exp_config["language"],
        "model_family": exp_config["model_family"],
        "seed": exp_config["seed"],
        "context": input_variant["context"],
        "features": ",".join(input_variant["features"]),
        "include_mwe_segment": input_variant["include_mwe_segment"],
        "transform": input_variant["transform"],
    }

    rows = []

    if isinstance(metrics, dict) and "overall" in metrics:
        overall_metrics = metrics.get("overall", {})
        overall_cm = overall_metrics.get("confusion_matrix_values", {})

        rows.append({
            **base,
            "eval_language": "overall",
            "accuracy": overall_metrics.get("accuracy"),
            "macro_f1": overall_metrics.get("macro_f1"),
            "macro_precision": overall_metrics.get("macro_precision"),
            "macro_recall": overall_metrics.get("macro_recall"),
            "tp": overall_cm.get("tp"),
            "tn": overall_cm.get("tn"),
            "fp": overall_cm.get("fp"),
            "fn": overall_cm.get("fn"),
        })

        for eval_lang, lang_metrics in metrics.get("per_language", {}).items():
            lang_cm = lang_metrics.get("confusion_matrix_values", {})
            rows.append({
                **base,
                "eval_language": str(eval_lang),
                "accuracy": lang_metrics.get("accuracy"),
                "macro_f1": lang_metrics.get("macro_f1"),
                "macro_precision": lang_metrics.get("macro_precision"),
                "macro_recall": lang_metrics.get("macro_recall"),
                "tp": lang_cm.get("tp"),
                "tn": lang_cm.get("tn"),
                "fp": lang_cm.get("fp"),
                "fn": lang_cm.get("fn"),
            })

    else:
        flat_metrics = metrics if isinstance(metrics, dict) else {}
        flat_cm = flat_metrics.get("confusion_matrix_values", {})

        rows.append({
            **base,
            "eval_language": str(exp_config.get("language")),
            "accuracy": flat_metrics.get("accuracy"),
            "macro_f1": flat_metrics.get("macro_f1"),
            "macro_precision": flat_metrics.get("macro_precision"),
            "macro_recall": flat_metrics.get("macro_recall"),
            "tp": flat_cm.get("tp"),
            "tn": flat_cm.get("tn"),
            "fp": flat_cm.get("fp"),
            "fn": flat_cm.get("fn"),
        })

    return rows



def load_all_runs(runs_root: Path) -> pd.DataFrame:
    all_rows: list[dict] = []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "experiment_config.json").exists():
            continue
        if not (d / "metrics.json").exists():
            continue
        all_rows.extend(flatten_run(d))
    return pd.DataFrame(all_rows)


def create_evaluation_overview(experiments_root, results_root) -> pd.DataFrame:
    df = load_all_runs(experiments_root)
    ensure_dir(results_root)

    """
    # Save master long table (single seed -> one row per run per eval_language)
    df.to_csv(out_dir / "master_metrics_long.csv", index=False)


    delta_highlight = ablation_delta(
        df,
        group_cols=["setting", "language_mode", "language", "model_family", "seed", "context", "transform", "include_mwe_segment"],
        baseline_filter={"features": {"contains": "empty"}},       
        variant_filter={"features": {"contains": "highlight"}},
        eval_languages=("overall", "EN", "PT", "GL"),
    )
    delta_highlight.to_csv(out_dir / "delta__highlight_vs_none.csv", index=False)


    one_shot_gain = ablation_delta(
        df,
        group_cols=["language_mode", "language", "model_family", "seed", "context", "features", "transform", "include_mwe_segment", "eval_language"],
        baseline_filter={"setting": "zero_shot"},
        variant_filter={"setting": "one_shot"},
        eval_languages=("overall", "EN", "PT", "GL"),
    )
    one_shot_gain.to_csv(out_dir / "delta__one_shot_minus_zero_shot.csv", index=False)

    # Per-signal view (overall) for quick inspection
    view_per_signal(df, eval_language="overall").to_csv(out_dir / "view__per_signal__overall.csv", index=False)
    view_per_signal(df, eval_language="EN").to_csv(out_dir / "view__per_signal__EN.csv", index=False)
    view_per_signal(df, eval_language="PT").to_csv(out_dir / "view__per_signal__PT.csv", index=False)

    print(f"[analysis] wrote outputs to: {out_dir}")
    """


def create_evaluation_plots(results_dir: Path) -> None:
    master = pd.read_csv(results_dir / "master_metrics_long.csv")
    delta_highlight = pd.read_csv(results_dir / "delta__highlight_vs_none.csv") if (results_dir / "delta__highlight_vs_none.csv").exists() else None
    one_shot = pd.read_csv(results_dir / "delta__one_shot_minus_zero_shot.csv") if (results_dir / "delta__one_shot_minus_zero_shot.csv").exists() else None

    plots_dir = results_dir / "plots"
    ensure_dir(plots_dir)


    contexts = ["previous_target_next", "target"]  
    signals = ["none", "highlight", "gloss+NER"]  

    for setting in sorted(master["setting"].dropna().unique()):
        for model_family in sorted(master["model_family"].dropna().unique()):
            
            plot_heatmap_context_signal(
                master, plots_dir,
                setting=setting, model_family=model_family,
                eval_language="overall",
                language_mode="multilingual",
                contexts=contexts, signals=signals
            )
            
            for ev in ["EN", "PT", "GL"]:
                plot_heatmap_context_signal(
                    master, plots_dir,
                    setting=setting, model_family=model_family,
                    eval_language=ev,
                    language_mode="multilingual",
                    contexts=contexts, signals=signals
                )

    
    if delta_highlight is not None:
        for ev in ["overall", "EN", "PT", "GL"]:
            plot_delta_bar(
                delta_highlight, plots_dir,
                title="Δ highlight vs none",
                out_name="delta__highlight_vs_none",
                eval_language=ev,
                group_by=["setting","model_family","context","language_mode","language"],
            )


    if one_shot is not None:
        for ev in ["overall", "EN", "PT", "GL"]:
            plot_one_shot_gain(one_shot, plots_dir, eval_language=ev)


    for setting in sorted(master["setting"].dropna().unique()):
        for model_family in sorted(master["model_family"].dropna().unique()):
            plot_en_pt_scatter(master, plots_dir, setting=setting, model_family=model_family)

    print(f"[plots] wrote plots to: {plots_dir}")


def run_evaluation():
    overview_df = create_evaluation_overview(experiments_root=PATHS.runs, results_root=PATHS.results)

