from pathlib import Path
from typing import Any, Dict

from analysis.evaluate_subslices import evaluate_subslices
from analysis_submodule.stress_masking import run_stress_masking_over_all_runs
from evaluation.run_evaluation import run_evaluation
from analysis_submodule.main_analysis_isolated import baseline_overview_table, context_signal_grouped_language_table, plot_context_connected_points, plot_context_impact_slope, plot_context_variant_heatmaps_per_model, plot_en_pt_gap_barplot, plot_one_shot_gains_baseline, plot_performance_heatmap, plot_f1_over_variants_4lines
from analysis_submodule.slice_analysis import compute_hard_control_gap_all_runs, hard_control_gap_for_run, plot_hard_control_gap_aggregated
from analysis_submodule.language_training_analysis import build_table_joint_vs_isolated, plot_joint_vs_isolated_connected, plot_regime_connected_big_figure, table3_joint_minus_isolated_deltas
from analysis_submodule.main_analysis_joint import plot_context_effect_by_regime, plot_en_pt_gap_by_regime, plot_heatmaps_context_variant_by_regime, table1_baseline_scoreboard_by_regime, table2_context_variant_by_language_joint
from analysis_submodule.utils.plots import plot_loss_curves_flat_comparison, plot_loss_curves_nested, plot_loss_curves_flat
from utils.helper import ensure_dir, read_json
from analysis_submodule.utils.helper import load_results_overviews,  save_multicol_latex, prepare_master_with_regime

   
def load_learning_curves(experiments_root: Path, model_family: str) -> Dict[str, Any]:
    baseline_dir_name = (
        f"zero_shot__EN__previous_target_next_True_none_empty__{model_family}__seed51"
    )
    learning_curves_path = experiments_root / baseline_dir_name / "learning_curves.json"
    return read_json(learning_curves_path)


def plot_baseline_loss_curves(experiments_root: Path, save_dir: Path) -> None:
    loss_curves = {
        "mBERT": load_learning_curves(experiments_root, "mBERT"),
        "logreg_tfidf": load_learning_curves(experiments_root, "logreg_tfidf"),
        "logreg_word2vec": load_learning_curves(experiments_root, "logreg_word2vec"),
    }

    plot_loss_curves_nested(loss_curves["mBERT"], save_dir, "mBERT")

    plot_loss_curves_flat_comparison(
        loss_curves["logreg_tfidf"],
        loss_curves["logreg_word2vec"],
        label_1="logreg_tfidf",
        label_2="logreg_word2vec",
        save_dir=save_dir,
    )




def run_deeper_analysis(experiments_root, results_root):
    results_sub_dir = results_root / "results_Michele"
    plots_path = results_sub_dir / "plots"
    ensure_dir(plots_path)

    plot_baseline_loss_curves(experiments_root, plots_path)

    # load aggregated results from experiments
    master_df, slices_df, masking_df = load_results_overviews(experiments_root, results_root, results_sub_dir)

    df_reg = prepare_master_with_regime(master_df, train_lang_joint="EN_PT_GL")

    t1 = table1_baseline_scoreboard_by_regime(df_reg)
    save_multicol_latex(t1, results_sub_dir, "table1__baseline_by_regime")

    t2 = table2_context_variant_by_language_joint(df_reg)
    for mf, tab in t2.items():
        save_multicol_latex(tab, results_sub_dir, f"table2__joint_context_variant__{mf}")

    t3 = table3_joint_minus_isolated_deltas(df_reg)
    for mf, tab in t3.items():
        save_multicol_latex(tab, results_sub_dir, f"table3__delta_joint_minus_isolated__{mf}")

    # ---- Usage ----
    table = build_table_joint_vs_isolated(master_df, model_family="mBERT", context="previous_target_next", variant="Standard")
    plot_joint_vs_isolated_connected(table, plots_path / "joint_vs_isolated__mBERT__standard_full.png")
    print(table.round(3).to_string(index=False))

    # 1) Big regime-connected figure (language × context facets)
    plot_regime_connected_big_figure(
        master_df,
        save_path=plots_path,
        model_family="mBERT",
        train_lang_joint="EN_PT_GL",
        eval_languages=("EN","PT","GL"),
    )

    # 2) “All plots but with regime”
    plot_context_effect_by_regime(
        master_df,
        save_path=plots_path / "context_effect_by_regime__mBERT.png",
        model_family="mBERT",
        train_lang_joint="EN_PT_GL",
        eval_languages=("EN","PT","GL"),
    )

    plot_heatmaps_context_variant_by_regime(
        master_df,
        save_dir=plots_path / "heatmaps_regime",
        model_family="mBERT",
        train_lang_joint="EN_PT_GL",
        eval_languages=("EN","PT","GL"),
    )

    plot_en_pt_gap_by_regime(
        master_df,
        save_path=plots_path / "gap_EN_PT_by_regime__mBERT.png",
        model_family="mBERT",
        train_lang_joint="EN_PT_GL",
    )
    
    plot_f1_over_variants_4lines(master_df, plots_path/"lines_variants__isolated.png",
                            model_family="mBERT", regime="isolated")

    plot_f1_over_variants_4lines(master_df, plots_path/"lines_variants__joint.png",
                                model_family="mBERT", regime="joint", train_lang_joint="EN_PT_GL")
    """
    tab1 = baseline_overview_table(master_df).round(3)
    save_multicol_latex(tab1, results_sub_dir, f"table1__baseline_scoreboard")
    tables_B = context_signal_grouped_language_table(master_df, setting="zero_shot")
    for mf, tab in tables_B.items():
        save_multicol_latex(tab, results_sub_dir, f"table2__full_target_delta__{mf}")

    
    plot_en_pt_gap_barplot(master_df, plots_path, setting="zero_shot")

    

    for mf in sorted(master_df["model_family"].dropna().unique()):
        plot_one_shot_gains_baseline(master_df, plots_path, model_family=mf)
        plot_context_connected_points(master_df, plots_path, setting="zero_shot", model_family=mf)
        plot_context_variant_heatmaps_per_model(master_df, plots_path, setting="zero_shot", model_family=mf)
        plot_performance_heatmap(master_df, plots_path, model_family=mf)
        plot_context_impact_slope(master_df, plots_path, model_family=mf)

    ###############################################

    hard_control_gap_for_run(slices_df, run_dir="zero_shot__EN__previous_target_next_True_highlight_ner__mBERT__seed51")
    gaps = compute_hard_control_gap_all_runs(slices_df)
    plot_hard_control_gap_aggregated(gaps, save_path=plots_path / "rq2__hard_control_gap__aggregated.png")
    
    # create plots
    #plot_masking_df_masking_deltas_bar(masking_df, master_df, plots_dir / "rq1rq2_masking_df_deltas_bar.png")
    #plot_masking_df_masking_scatter(results_root, plots_dir / "rq2_masking_df_scatter.png")
    #plot_hard_control_delta(results_root, plots_dir / "rq2_hard_control_delta.png", aggregate=True)

"""





    """
    import numpy as np
from sklearn.metrics import f1_score
from sklearn.utils import resample

# Load your predictions file (NOT the master metrics, the one with row-level preds)
# preds_df = pd.read_csv('test_predictions.csv') 

def bootstrap_f1(g, n_boot=1000):
    scores = []
    y_true = g['label'].values
    y_pred = g['test_pred'].values # or whatever your pred col is named
    
    for _ in range(n_boot):
        # Sample indices with replacement
        indices = resample(range(len(y_true)), replace=True)
        score = f1_score(y_true[indices], y_pred[indices], average='macro')
        scores.append(score)
    
    mean_score = np.mean(scores)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    
    return pd.Series({
        'macro_f1': mean_score, 
        'ci_lower': lower, 
        'ci_upper': upper
    })

# Group by your experiment config
# This might take a minute or two to run!
results_with_ci = preds_df.groupby(
    ['language', 'model_family', 'setting', 'variant', 'context']
).apply(bootstrap_f1).reset_index()

# Now you can create a table with "0.75 (0.73–0.77)"
results_with_ci['report_string'] = results_with_ci.apply(
    lambda x: f"{x['macro_f1']:.2f} ({x['ci_lower']:.2f}–{x['ci_upper']:.2f})", 
    axis=1
)

print(results_with_ci[['language', 'variant', 'report_string']])
    
    """