from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from analysis.evaluate_subslices import evaluate_subslices
from analysis_submodule.stress_masking import run_stress_masking_over_all_runs
from analysis_submodule.utils.plots import load_base_from_master, plot_delta_bar, plot_en_pt_scatter, plot_heatmap_context_signal, plot_one_shot_gain, rq1_plot_surface_reliance, rq2_plot_hard_control_delta, rq2_plot_masking_df_scatter
from evaluation.run_evaluation import run_evaluation
from utils.helper import ensure_dir, read_csv_data



def clean_data(df):
    df = df.copy()
    
    # 1. Fill NaNs in features/transform
    df['features'] = df['features'].fillna('')
    df['transform'] = df['transform'].fillna('none')
    
    # 2. Create readable Variant names
    def get_variant(row):
        parts = []
        if str(row['transform']) == 'highlight': parts.append("Highlight")
        
        feats = str(row['features'])
        if 'ner' in feats: parts.append("NER")
        if 'gloss' in feats: parts.append("Glosses")
        
        if not parts: return "Standard"
        return " + ".join(parts)
    
    df['variant'] = df.apply(get_variant, axis=1)
    
    # 3. Rename Contexts
    df['context_label'] = df['context'].replace({
        'previous_target_next': 'Full Context',
        'target': 'Target Only',
        'all': 'Full Context'
    })
    
    return df



def ai_studio(df):


    # Apply cleaning
    df_clean = clean_data(df)

    # Filter for Zero Shot mBERT (The main experiment set)
    zero_shot_df = df_clean[
        (df_clean['setting'] == 'zero_shot') & 
        (df_clean['model_family'] == 'mBERT')
    ]

    # --- PLOT A: The Performance Heatmap (Overview) ---
    pivot = zero_shot_df.pivot_table(
        index='variant', 
        columns=['language', 'context_label'], 
        values='macro_f1'
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot, 
        annot=True, 
        fmt=".3f", 
        cmap="RdYlGn", 
        center=0.75, # Center color map around decent performance
        linewidths=.5,
        cbar_kws={'label': 'Macro F1'}
    )
    plt.title("mBERT Zero-Shot Performance: Features vs. Context", fontsize=14)
    plt.ylabel("Input Variant")
    plt.xlabel("Language / Context")
    plt.tight_layout()
    plt.show()

    # --- PLOT B: Context Impact Slope Chart ---
    # Shows the weird EN behavior vs the normal PT behavior
    g = sns.catplot(
        data=zero_shot_df,
        x='context_label', 
        y='macro_f1',
        hue='variant',
        col='language',
        kind='point',
        height=5, 
        aspect=1.0,
        palette='tab10',
        markers='o',
        order=['Target Only', 'Full Context'] # Enforce order
    )
    g.fig.suptitle("Does Context Help? (Slope Analysis)", y=1.05)
    g.set_axis_labels("", "Macro F1 Score")
    plt.show()

    # --- PLOT C: One-Shot Gains ---
    # Compare Standard/Full/Zero vs Standard/Full/One
    comparison = df_clean[
        (df_clean['variant'] == 'Standard') & 
        (df_clean['context_label'] == 'Full Context')
    ].copy()

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=comparison,
        x='language',
        y='macro_f1',
        hue='setting',
        palette=['#95a5a6', '#2ecc71'], # Grey, Green
        edgecolor='black'
    )

    # Annotate bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)

    plt.title("Impact of One-Shot Training (mBERT Standard)", fontsize=13)
    plt.ylabel("Macro F1")
    plt.ylim(0, 1.05)
    plt.legend(title='Setting', loc='upper left')
    plt.tight_layout()
    plt.show()


def chatty(df):
    pass

    


def load_results_overviews(experiments_root, results_root, results_sub_dir, split_type="test"):
    master_csv_path = results_root / "master_metrics_long.csv"
    if not master_csv_path.exists():
        run_evaluation(split_type=split_type)
    master_df = read_csv_data(master_csv_path)

    slice_csv_path = results_root / "slices_overview.csv"
    if not slice_csv_path.exists():
        evaluate_subslices(split_type=split_type)
    slices_df = read_csv_data(slice_csv_path)

    masking_csv_path = results_sub_dir / "stress_masking_summary.csv"
    if not masking_csv_path.exists():
        run_stress_masking_over_all_runs(experiments_root, results_root)
    masking_df = read_csv_data(masking_csv_path)

    return master_df, slices_df, masking_df


def create_evaluation_plots(experiments_root, results_root: Path, results_sub_dir) -> None:
    plots_dir = results_sub_dir / "plots"
    ensure_dir(plots_dir)

    # load aggregated results from experiments
    master_df, slices_df, masking_df = load_results_overviews(experiments_root, results_root, results_sub_dir)
    base = load_base_from_master(master_df)

    # Optional delta tables
    delta_highlight_path = results_sub_dir / "delta__highlight_vs_none.csv"
    delta_one_shot_path = results_sub_dir / "delta__one_shot_minus_zero_shot.csv"
    delta_highlight = read_csv_data(delta_highlight_path) if delta_highlight_path.exists() else None
    one_shot = read_csv_data(delta_one_shot_path) if delta_one_shot_path.exists() else None

    # ---- Heatmaps (context × signal) ----
    contexts = ["previous_target_next", "target"]
    signals = [
        "empty",
        "highlight",
        "glosses",
        "highlight+glosses",
        "ner",
        "highlight+ner",
        "glosses,ner",
        "highlight+glosses,ner",
    ]

    for setting in sorted(master_df["setting"].dropna().unique()):
        for model_family in sorted(master_df["model_family"].dropna().unique()):
            plot_heatmap_context_signal(
                master_df,
                plots_dir,
                setting=setting,
                model_family=model_family,
                eval_language="overall",
                language_mode="multilingual",
                contexts=contexts,
                signals=signals,
            )
            for ev in ["EN", "PT", "GL"]:
                plot_heatmap_context_signal(
                    master_df,
                    plots_dir,
                    setting=setting,
                    model_family=model_family,
                    eval_language=ev,
                    language_mode="multilingual",
                    contexts=contexts,
                    signals=signals,
                )

    # ---- RQ1 + RQ2 (masking_df masking) ----
    rq1_plot_surface_reliance(masking_df, base, plots_dir / "rq1_surface_reliance.png")
    rq2_plot_masking_df_scatter(masking_df, base, plots_dir / "rq2_masking_df_scatter.png")

    # ---- RQ2 support (hard vs control) ----
    rq2_plot_hard_control_delta(
        slices_df, base, plots_dir / "rq2_hard_control_gap.png",
        hard_label="hard", control_label="control", aggregate=True
    )

    # ---- Existing delta plots (if available) ----
    if delta_highlight is not None:
        for ev in ["overall", "EN", "PT", "GL"]:
            plot_delta_bar(
                delta_highlight,
                plots_dir / f"delta__highlight_vs_none__eval-{ev}.png",
                title="Δ highlight vs none",
                eval_language=ev,
                group_by=["setting", "model_family", "context", "language_mode", "language"],
            )

    if one_shot is not None:
        for ev in ["overall", "EN", "PT", "GL"]:
            plot_one_shot_gain(
                one_shot,
                plots_dir / f"delta__one_shot_gain__eval-{ev}.png",
                eval_language=ev,
            )

    # ---- EN vs PT scatter ----
    for setting in sorted(master_df["setting"].dropna().unique()):
        for model_family in sorted(master_df["model_family"].dropna().unique()):
            plot_en_pt_scatter(
                master_df,
                plots_dir / f"scatter__EN_vs_PT__{setting}__{model_family}.png",
                setting=setting,
                model_family=model_family,
            )

    print(f"[plots] wrote plots to: {plots_dir}")
    

def run_further_analysis(experiments_root, results_root):
    results_sub_dir = results_root / "results_Michele"

    #create_evaluation_plots(experiments_root, results_root, results_sub_dir)

    # load aggregated results from experiments
    master_df, slices_df, masking_df = load_results_overviews(experiments_root, results_root, results_sub_dir)

    ai_studio(master_df)
    #chatty(master_df)

    # create plots
    #rq1_plot_surface_reliance(masking_df, master_df, plots_dir)
    #plot_masking_df_masking_deltas_bar(masking_df, master_df, plots_dir / "rq1rq2_masking_df_deltas_bar.png")
    #plot_masking_df_masking_scatter(results_root, plots_dir / "rq2_masking_df_scatter.png")
    #plot_hard_control_delta(results_root, plots_dir / "rq2_hard_control_delta.png", aggregate=True)


    # deltas vs ALL
    #if include_all_reference:
    #    df_with_deltas = add_deltas_vs_reference(df_long_all, ref_slice=all_slice_name)
    #    df_with_deltas.to_csv(save_dir / f"slice_metrics_with_deltas_vs_{all_slice_name}.csv", index=False)
    #else:
    #    df_with_deltas = df_long_all

    # optional: hard vs control deltas (if you have these slice names)
    # common names from your pipeline:
    #   slice_ambiguous == "hard"/"control" would need to be turned into ID lists if you want them here.
    # If you stored them as ID lists in slice_ids.json, then these will exist.
    #if "ambiguous_mwe_ids" in slice_ids and "control_ids" in slice_ids:
        # you can create these keys in your slice-ids creation step if desired
    #    pass




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