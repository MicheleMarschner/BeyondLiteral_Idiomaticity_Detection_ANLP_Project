from pathlib import Path

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt



from analysis_submodule.analysis_Michele.utils.helper_analysis import CONTEXT_ORDER, VARIANT_ORDER, assert_unique, save_plot


def plot_stress_masking_lines_monolingual(
    df_mask: pd.DataFrame,
    save_path: Path,
    title: str,
    model_family: str = "mBERT",
) -> pd.DataFrame:
    """
    Two-panel stress-masking line plot for monolingual runs:
      panels = eval_language (EN/PT)
      hue    = context_label (Full/Target)
      style  = masking condition (global single vs targeted context)
      x      = variant
      y      = delta Macro-F1 vs unmasked
    """
    df = df_mask.copy()
    df = df[df["model_family"].astype(str) == str(model_family)].copy()

    value_cols = [
        "delta_macro_f1_global_single_mask",
        "delta_macro_f1_targeted_context_mask",
    ]

    df = df.dropna(
        subset=["eval_language", "context_label", "variant"] + value_cols
    ).copy()
    if df.empty:
        return df

    assert_unique(
        df,
        keys=["eval_language", "context_label", "variant"],
        what="plot_stress_masking_lines_monolingual uniqueness",
    )

    long = df.melt(
        id_vars=["eval_language", "context_label", "variant"],
        value_vars=value_cols,
        var_name="mask_variant",
        value_name="delta_macro_f1",
    )

    mask_label_map = {
        "delta_macro_f1_global_single_mask": "Global single mask",
        "delta_macro_f1_targeted_context_mask": "Targeted context mask",
    }
    mask_order = ["Global single mask", "Targeted context mask"]

    long["mask_variant"] = long["mask_variant"].map(mask_label_map)

    long["variant"] = pd.Categorical(
        long["variant"].astype(str),
        categories=VARIANT_ORDER,
        ordered=True,
    )
    long["context_label"] = pd.Categorical(
        long["context_label"].astype(str),
        categories=CONTEXT_ORDER,
        ordered=True,
    )
    long["eval_language"] = pd.Categorical(
        long["eval_language"].astype(str),
        categories=["EN", "PT"],
        ordered=True,
    )
    long["mask_variant"] = pd.Categorical(
        long["mask_variant"].astype(str),
        categories=mask_order,
        ordered=True,
    )

    long = long.sort_values(
        ["eval_language", "context_label", "mask_variant", "variant"]
    ).reset_index(drop=True)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)

    y_min = min(-0.05, float(long["delta_macro_f1"].min()) * 1.08)
    y_max = max(0.02, float(long["delta_macro_f1"].max()) * 1.08)

    for ax, lang in zip(axes, ["EN", "PT"]):
        sub = long[long["eval_language"].astype(str) == lang].copy()
        if sub.empty:
            continue

        sns.lineplot(
            data=sub,
            x="variant",
            y="delta_macro_f1",
            hue="context_label",
            style="mask_variant",
            markers=True,
            dashes=True,
            linewidth=2,
            estimator=np.mean,  # NO-OP due to assert_unique
            errorbar=None,
            ax=ax,
        )

        ax.axhline(0, color="gray", linewidth=1)
        ax.set_title(lang)
        ax.set_xlabel("Input variant")
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(y_min, y_max)

    axes[0].set_ylabel("Δ Macro-F1 vs unmasked")
    axes[1].set_ylabel("")

    left_legend = axes[0].get_legend()
    if left_legend is not None:
        left_legend.remove()

    handles, labels = axes[1].get_legend_handles_labels()
    right_legend = axes[1].get_legend()
    if right_legend is not None:
        right_legend.remove()

    axes[1].legend(
        handles,
        labels,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        title="",
    )

    fig.suptitle(title, y=1.03)

    save_plot(fig, save_path)
    return long