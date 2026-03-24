from pathlib import Path
import pandas as pd
from typing import Any, Dict

from analysis_submodule.analysis_Michele.monolingual_vs_multilingual_analysis import plot_multi_vs_mono_delta_bars_by_model_family, plot_delta_train_over_variants_grid, plot_training_setup_connected_big_figure, tables_train_delta_context_variant
from analysis_submodule.analysis_Michele.cross_lingual_analysis import plot_pt_transfer_class_recall, plot_pt_transfer_comparison
from analysis_submodule.analysis_Michele.input_signals_analysis import plot_en_pt_gap, plot_input_variants_lines_per_lan_setup, plot_joint_over_variants, table2_context_variant
from analysis_submodule.analysis_Michele.one_shot_analysis import run_one_shot_experiment
from analysis_submodule.analysis_Michele.utils.helper_analysis import EVAL_ORDER_ISO, EVAL_ORDER_MULTI, create_folder_structure, pivot_strict, save_multicol_latex
from analysis_submodule.analysis_Michele.utils.plots import plot_loss_curves_flat_comparison, plot_loss_curves_nested
from analysis_submodule.analysis_Michele.utils.data_views import filter_baseline, get_data_for_setup, load_results_overviews, load_stress_masking_monolingual, summarize_global_n_vs_single_gap
from analysis_submodule.analysis_Michele.stress_masking_analysis import plot_stress_masking_lines_monolingual
from utils.helper import ensure_dir, read_json



def create_main_results_table(
    master_df: pd.DataFrame,
    setting: str = "zero_shot",
) -> pd.DataFrame:
    """Main results table: language on top, mono/multi/delta underneath."""
    df_mono = get_data_for_setup(
        master_df,
        setup="monolingual",
        setting=setting,
    )
    df_multi = get_data_for_setup(
        master_df,
        setup="multilingual",
        setting=setting,
    )

    df = pd.concat([df_mono, df_multi], ignore_index=True)
    df = filter_baseline(df)

    tab = pivot_strict(
        df,
        index=["model_family"],
        columns=["eval_language", "training_setup"],
        values="macro_f1",
        what="table_main_results",
    )

    out = pd.DataFrame(index=tab.index)

    for lang in ["EN", "PT"]:
        mono_col = (lang, "monolingual")
        multi_col = (lang, "multilingual")

        if mono_col in tab.columns:
            out[(lang, "mono")] = tab[mono_col]
        if multi_col in tab.columns:
            out[(lang, "multi")] = tab[multi_col]
        if mono_col in tab.columns and multi_col in tab.columns:
            out[(lang, "DELTA_TMP")] = tab[multi_col] - tab[mono_col]

    col_order = [
        ("EN", "mono"), ("EN", "multi"), ("EN", "DELTA_TMP"),
        ("PT", "mono"), ("PT", "multi"), ("PT", "DELTA_TMP"),
    ]
    col_order = [c for c in col_order if c in out.columns]

    out = out.reindex(columns=col_order)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index.name = None
    return out


# -----------------------------------------------------------------------------
# train_val loss curves !TODO Anpassung für ggf. macro f1?
# -----------------------------------------------------------------------------
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



# -----------------------------------------------------------------------------
# main function
# -----------------------------------------------------------------------------
def run_analysis(experiments_root: Path, results_root: Path) -> None:
    #setting = "zero_shot"
    results_sub_dir = results_root / "results_Michele"
    tables_path, plots_path = create_folder_structure(results_sub_dir)
    setting = "zero_shot"

    #plot_baseline_loss_curves(experiments_root, plots_path)

    # load aggregated results from experiments 
    master_df, slices_df = load_results_overviews(experiments_root, results_root, results_sub_dir)
    masking_mono_df = load_stress_masking_monolingual(results_sub_dir)

    # create filtered view for main experiment analysis
    master_filtered_df = master_df[
        ~(
            (master_df["run_dir"] == "zero_shot__EN__previous_target_next_True_none_empty__modernBERT__seed51")
            | (master_df["language_mode"] == "cross_lingual")
        )
    ].copy()


    # main table baseline - model family
    df_mono = get_data_for_setup(master_df, setup="monolingual", setting=setting)
    df_multi = get_data_for_setup(master_df, setup="multilingual", setting=setting)

    df_setup = pd.concat([df_mono, df_multi], ignore_index=True)
    df_setup = filter_baseline(df_setup)

    tab_main = create_main_results_table(master_df, setting=setting)
    
    save_multicol_latex(tab_main, tables_path, "table__main_results__baseline", decimals=3)


    ### Context, Signals and language regimes
    plot_multi_vs_mono_delta_bars_by_model_family(
        master_filtered_df,
        save_dir=plots_path / "delta_multi_vs_mono_bars",
        setting="zero_shot",
    )

    for setup, eval_order in [("multilingual", EVAL_ORDER_MULTI), ("monolingual", EVAL_ORDER_ISO)]:
        df_setup = get_data_for_setup(master_filtered_df, setup=setup, setting=setting)

        for mf in sorted(df_setup["model_family"].dropna().unique()):
            df_mf = df_setup[df_setup["model_family"] == mf].copy()
            out_dir = results_sub_dir / setup
            ensure_dir(out_dir)

            """
            ### !TODO fliegt eher raus wenn nicht appendix
            variant_table = table2_context_variant(df_mf, eval_order=eval_order)
            for k, tab in variant_table.items():
                save_multicol_latex(tab, out_dir, f"table_context_variant__{k}__{setup}", decimals=3)
            """

            plot_input_variants_lines_per_lan_setup(
                df_mf,
                out_dir / f"plot_variants_lines_{mf}.png",
                title=f"{setup.title()} | {mf} | Macro-F1 over variants ({setting})",
                eval_order=eval_order,
            )

            ### Stress masking diagnostic (monolingual | mBERT)
            gap_summary = summarize_global_n_vs_single_gap(masking_mono_df, model_family=mf)
            gap_summary.to_csv(
                tables_path / f"table__stress_masking_global_n_vs_single_gap__{mf}_mono.csv",
                index=False,
            )

            plot_stress_masking_lines_monolingual(
                masking_mono_df,
                plots_path / f"plot__stress_masking_lines__{mf}__monolingual.png",
                title=f"Monolingual | {mf} | Stress masking deltas over variants",
                model_family=mf,
            )

            """
            # EN–PT gap
            if set(["EN", "PT"]).issubset(set(df_mf["eval_language"].astype(str))):
                plot_en_pt_gap(
                    df_mf,
                    out_dir / "plot_en_pt_gap.png",
                    title=f"{setup.title()} | {mf} | EN–PT gap ({setting})",
                )
            """

    for setup, eval_order in [("multilingual", EVAL_ORDER_MULTI), ("monolingual", EVAL_ORDER_ISO)]:
        #df_setup = get_data_for_setup(master_df, setup=setup, setting=setting)
        run_one_shot_experiment(master_filtered_df, results_sub_dir, setup, train_lang_joint = "EN_PT_GL", eval_languages = ("EN", "PT"))

    
    ### context and input variants experiment
    plot_delta_train_over_variants_grid(
        master_filtered_df,
        save_path=plots_path / "delta_train_over_variants__grid__zero_shot.png",
        title="Δtrain (multilingual − monolingual) over variants | EN/PT | rows=context | cols=model_family",
        setting="zero_shot",
    )


    ### cross lingual transfer experiment
    df_transfer = plot_pt_transfer_comparison(
        master_df=master_df,
        save_path=plots_path / "pt_transfer_comparison_baseline.png",
        title="Portuguese test performance across training conditions",
        model_family="mBERT",
    )

    df_recall = plot_pt_transfer_class_recall(
        master_df=master_df,
        save_path=plots_path / "pt_transfer_class_recall.png",
        model_family="mBERT",
    )

    
    
    
    """
    ### TODO wird eher rausfliegen wenn nicht appendix
    tabs = tables_train_delta_context_variant(master_filtered_df, setting="zero_shot")
    for mf, t in tabs.items():
        safe_mf = str(mf).replace("/", "_").replace(" ", "_")
        save_multicol_latex(t, tables_path, f"table_train_delta_ctx_variant__{safe_mf}__zero_shot", decimals=3)

   
    ### TODO wird eher rausfliegen wenn nicht appendix
    for mf in sorted(master_df["model_family"].dropna().unique()):
        
        plot_training_setup_connected_big_figure(
            master_df,
            plots_path / f"plot_train_diff_big_overview__{mf}__zero_shot.png",
            model_family=mf,
            setting="zero_shot",
            eval_languages=("EN", "PT"),
        )
    """
    
    