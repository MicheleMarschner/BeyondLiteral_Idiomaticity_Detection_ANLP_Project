
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


from analysis_submodule.utils.helper import normalize_variant, save_plot


# ----------------------------
# RQ2: Stress masking scatter (normal vs mask_both)
# ----------------------------
def rq2_plot_stress_scatter(
    df_stress: pd.DataFrame,
    df_base: pd.DataFrame,
    out_path: Path,
    *,
    metric_x: str = "macro_f1_normal",
    metric_y: str = "macro_f1_both",
    hue: str = "setting",
    style: str = "signal",
) -> None:
    """
    Scatter plot: performance under normal inference vs masked (both) inference.
    Joins run metadata (context/features/transform) from df_base and builds `signal`.
    """
    required_stress = {"run_dir", metric_x, metric_y, hue}
    missing = required_stress - set(df_stress.columns)
    if missing:
        raise ValueError(f"df_stress missing columns: {sorted(missing)}")

    required_base = {"run_dir", "context", "features", "transform"}
    missing = required_base - set(df_base.columns)
    if missing:
        raise ValueError(f"df_base missing columns: {sorted(missing)}")

    join_base = df_base[list(required_base)].drop_duplicates(subset=["run_dir"]).copy()
    join_base = normalize_features(join_base)

    df = df_stress.merge(join_base, on="run_dir", how="left")
    df = normalize_features(df)
    df = add_signal_col(df)  # adds df["signal"]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.scatterplot(
        data=df,
        x=metric_x,
        y=metric_y,
        hue=hue,
        style=style,
        ax=ax,
        alpha=0.9,
    )

    lo = float(min(df[metric_x].min(), df[metric_y].min()))
    hi = float(max(df[metric_x].max(), df[metric_y].max()))
    ax.plot([lo, hi], [lo, hi], linewidth=1)

    ax.set_xlabel(metric_x.replace("_", " "))
    ax.set_ylabel(metric_y.replace("_", " "))
    ax.set_title("Stress masking: normal vs mask_both")

    _save_mpl(fig, out_path)




def rq2_plot_masking_df_scatter(
    masking_df: pd.DataFrame,
    base: pd.DataFrame,
    out_path: Path,
) -> None:
    # join context/features/transform for styling (masking_df already has setting/language/etc.)
    join_base = base[["run_dir", "context", "features", "transform"]].drop_duplicates(subset=["run_dir"]).copy()
    join_base["features"] = join_base["features"].fillna("").replace({"": "empty"})

    df = masking_df.merge(join_base, on="run_dir", how="left")
    df["features"] = df["features"].fillna("").replace({"": "empty"})

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.scatterplot(
        data=df,
        x="macro_f1_normal",
        y="macro_f1_both",
        hue="setting",
        style="features",
        ax=ax,
        alpha=0.9,
    )

    lo = float(min(df["macro_f1_normal"].min(), df["macro_f1_both"].min()))
    hi = float(max(df["macro_f1_normal"].max(), df["macro_f1_both"].max()))
    ax.plot([lo, hi], [lo, hi], linewidth=1)

    ax.set_xlabel("macro-F1 (normal)")
    ax.set_ylabel("macro-F1 (mask both)")
    ax.set_title("RQ2: normal vs mask_both (stress masking)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# Confusion matrix from counts (optional utility)
# ----------------------------
def plot_confusion_matrix_from_counts(
    cm_vals: Dict[str, int],
    out_path: Path,
    labels: Tuple[str, str] = ("Idiom", "Literal"),
    title: str = "Confusion Matrix",
) -> None:
    cm = np.array([[cm_vals["tn"], cm_vals["fp"]],
                   [cm_vals["fn"], cm_vals["tp"]]], dtype=float)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_row = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_row, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(title=title + " (row % + counts)", xlabel="Predicted", ylabel="True")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels([labels[0], labels[1]])
    ax.set_yticklabels([labels[0], labels[1]])

    thr = cm_row.max() / 2.0 if cm_row.size else 0
    for i in range(2):
        for j in range(2):
            count = int(cm[i, j])
            perc = cm_row[i, j] * 100
            color = "white" if cm_row[i, j] > thr else "black"
            ax.text(j, i, f"{count}\n{perc:.1f}%", ha="center", va="center", color=color)

    ax.set_ylim(1.5, -0.5)
    _save_mpl(fig, out_path)






# ----------------------------
# RQ2: stress masking scatter (normal vs both)
# ----------------------------
def rq2_plot_stress_scatter(
    df_stress: pd.DataFrame,
    df_base: pd.DataFrame,
    out_path: Path,
) -> None:
    join_base = df_base[["run_dir", "context", "features", "transform"]].copy()
    join_base = _normalize_features(join_base)

    df = df_stress.merge(join_base, on="run_dir", how="left")
    df = _normalize_features(df)
    df = add_signal_col(df)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.scatterplot(
        data=df,
        x="macro_f1_normal",
        y="macro_f1_both",
        hue="setting",
        style="signal",
        ax=ax,
        alpha=0.9,
    )

    lo = float(min(df["macro_f1_normal"].min(), df["macro_f1_both"].min()))
    hi = float(max(df["macro_f1_normal"].max(), df["macro_f1_both"].max()))
    ax.plot([lo, hi], [lo, hi], linewidth=1)

    ax.set_xlabel("macro-F1 (normal)")
    ax.set_ylabel("macro-F1 (mask both)")
    ax.set_title("Stress masking: normal vs mask_both")

    _save_mpl(fig, out_path)

