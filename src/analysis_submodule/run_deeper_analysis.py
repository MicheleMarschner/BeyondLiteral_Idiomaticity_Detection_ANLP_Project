from pathlib import Path
from typing import Any, Dict, Tuple

#from analysis.evaluate_subslices import evaluate_subslices
#from analysis_submodule.stress_masking import run_stress_masking_all
#from evaluation.run_evaluation import run_evaluation
#from analysis_submodule.slice_analysis import compute_hard_control_gap_all_runs, hard_control_gap_for_run, plot_hard_control_gap_aggregated
from analysis_submodule.language_training_analysis import plot_delta_train_bars_g2_per_model_family, plot_delta_train_over_variants_grid, plot_language_setup_connected_big_figure, table_train_delta_baseline, tables_train_delta_context_variant
from analysis_submodule.utils.plots import plot_loss_curves_flat_comparison, plot_loss_curves_nested
from utils.helper import read_json
from analysis_submodule.utils.helper import load_results_overviews, create_folder_structure, save_multicol_latex
from analysis_submodule.main_analysis import run_analysis

   
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


def run_deeper_analysis(experiments_root: Path, results_root: Path) -> None:
    results_sub_dir = results_root / "results_Michele"
    
    tables_path, plots_path = create_folder_structure(results_sub_dir)
    plot_baseline_loss_curves(experiments_root, plots_path)

    # load aggregated results from experiments
    master_df, slices_df = load_results_overviews(experiments_root, results_root, results_sub_dir)

    plot_delta_train_over_variants_grid(
        master_df,
        save_path=plots_path / "delta_train_over_variants__grid__zero_shot.png",
        title="Δtrain (joint − isolated) over variants | EN/PT | rows=context | cols=model_family",
        setting="zero_shot",
    )

    plot_delta_train_bars_g2_per_model_family(
        master_df,
        save_dir=plots_path / "delta_train_bars_g2",
        setting="zero_shot",
    )


    ## create main tables and plots
    run_analysis(master_df, results_sub_dir)

    tab = table_train_delta_baseline(master_df, setting="zero_shot")
    save_multicol_latex(tab, tables_path, "table_train_delta_baseline__zero_shot", decimals=3)

    tabs = tables_train_delta_context_variant(master_df, setting="zero_shot")
    for mf, t in tabs.items():
        safe_mf = str(mf).replace("/", "_").replace(" ", "_")
        save_multicol_latex(t, tables_path, f"table_train_delta_ctx_variant__{safe_mf}__zero_shot", decimals=3)

    for mf in sorted(master_df["model_family"].dropna().unique()):
        
        plot_language_setup_connected_big_figure(
            master_df,
            plots_path / f"plot_train_diff_big_overview__{mf}__zero_shot.png",
            model_family=mf,
            setting="zero_shot",
            eval_languages=("EN", "PT"),
        )


    """
    
    t3 = table3_joint_minus_isolated_deltas(df_lang_setup)
    for mf, tab in t3.items():
        save_multicol_latex(tab, tables_path, f"table3__delta_joint_minus_isolated__{mf}")

    # ---- Usage ----
    table = build_table_joint_vs_isolated(master_df, model_family="mBERT", context="previous_target_next", variant="Standard")
    plot_joint_vs_isolated_connected(table, plots_path / "joint_vs_isolated__mBERT__standard_full.png")
    print(table.round(3).to_string(index=False))

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