from pathlib import Path
import pandas as pd

from src.config import PATHS
from src.evaluation.plots import plot_delta_bar, plot_en_pt_scatter, plot_heatmap_context_signal, plot_one_shot_gain
from src.evaluation.reporting import ablation_delta, load_all_runs, view_per_signal
from src.utils.helper import ensure_dir


def create_evaluation_tables_and_views() -> None:
    df = load_all_runs(PATHS.runs)

    out_dir = PATHS.results
    ensure_dir(out_dir)

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
    create_evaluation_tables_and_views()
    # from src.config import PATHS
    # stats_long = load_split_stats_table(PATHS.runs)
    # stats_long.to_csv(PATHS.results / "split_stats_long.csv", index=False)
    # paper_stats = make_paper_data_stats(stats_long)
    # paper_stats.to_csv(PATHS.results / "data_stats_table.csv", index=False)
    # print(paper_stats)
    #print(paper_stats[paper_stats["language"].astype(str).str.contains("EN")])
    create_evaluation_plots(PATHS.results)


