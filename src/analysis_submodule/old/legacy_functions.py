
# -----------------------------------------------------------------------------
# RQ3: Heatmap + connected line plot (context × variant) — per model_family
# -----------------------------------------------------------------------------

from analysis_submodule.main_analysis import LANGUAGE_SETUP_ORDER, VARIANT_ORDER, save_plot


def plot_context_effect(
    df_setup: pd.DataFrame,
    save_path: Path,
    *,
    title: str,
    eval_order: list[str],
    metric: str = "macro_f1",
) -> None:
    """
    Connected points (Full vs Target), faceted by eval_language, hue=variant.

    We keep strict uniqueness (so mean is a no-op), but use estimator="mean"
    because seaborn still aggregates internally for point plots.
    Also remove unused categories to avoid empty facets triggering seaborn/pandas errors.
    """
    df = df_setup.copy()

    # keep only rows that can be plotted (avoids seaborn empty-group bug)
    df = df.dropna(subset=["eval_language", "variant", "context_label", metric]).copy()
    if df.empty:
        return

    # strict: must be unique per plotted point (otherwise seaborn would average)
    assert_unique(
        df,
        keys=["eval_language", "variant", "context_label"],
        what="plot_context_effect cell uniqueness",
    )

    # ensure ordered categoricals but ONLY for categories that exist
    present_langs = [l for l in eval_order if l in set(df["eval_language"].astype(str))]
    df["eval_language"] = pd.Categorical(df["eval_language"].astype(str), categories=present_langs, ordered=True)

    df["variant"] = pd.Categorical(df["variant"].astype(str), categories=VARIANT_ORDER, ordered=True)
    df["context_label"] = pd.Categorical(df["context_label"].astype(str), categories=CONTEXT_ORDER, ordered=True)

    # remove unused categories (critical to avoid empty facets/groups)
    df["eval_language"] = df["eval_language"].cat.remove_unused_categories()
    df["variant"] = df["variant"].cat.remove_unused_categories()
    df["context_label"] = df["context_label"].cat.remove_unused_categories()

    g = sns.catplot(
        data=df,
        x="context_label",
        y=metric,
        hue="variant",
        col="eval_language",
        col_order=present_langs,
        kind="point",
        estimator=np.mean,   # mean is a NO-OP because we asserted uniqueness
        errorbar=None,
        height=3.5,
        aspect=0.95,
        dodge=True,
        markers="o",
        order=CONTEXT_ORDER,
        hue_order=VARIANT_ORDER,
        linestyles="-",
    )
    g.set_axis_labels("", "Macro-F1")
    g.set_titles("Eval: {col_name}")
    g.fig.suptitle(title, y=1.05)

    save_plot(g.fig, save_path)


def plot_context_variant_heatmaps_per_model(
    master_df: pd.DataFrame,
    save_dir: Path,
    setting: str = "zero_shot",
    model_family: str = "mBERT",
    include_mwe_segment: bool = True,
    language_setup: str = "isolated",
    eval_languages: list[str] | None = None,
    train_lang_joint: str = "EN_PT_GL",
    figsize: tuple[int, int] = (7, 2),
) -> None:
    '''
    For each model_family: draw one heatmap per eval_language.
    Heatmap axes: rows=context (Full/Target), cols=variant, value=macro-F1.
    '''
    df = prepare_master_with_language_setup(
        master_df,
        setting=setting,
        model_family=model_family,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=("EN", "PT", "GL", "Joint"),
    )

    df = df[df["language_setup"] == language_setup].copy()
    if df.empty:
        print(f"[heatmap] No data for model_family={model_family}, language_setup={language_setup}, setting={setting}")
        return

    if eval_languages is None:
        eval_languages = [l for l in ["EN", "PT", "GL", "Joint"] if l in set(df["eval_language"].astype(str))]

    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    df["context_label"] = pd.Categorical(df["context_label"], categories=CONTEXT_ORDER, ordered=True)
    
    for lang in eval_languages:
        sub = df[df["eval_language"].astype(str) == lang].copy()
        if sub.empty:
            continue

        heat = sub.pivot_table(
            index="context_label",
            columns="variant",
            values="macro_f1",
            aggfunc="first",
        ).reindex(index=CONTEXT_ORDER)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            heat,
            annot=True,
            fmt=".3f",
            cbar=True,
            linewidths=0.5,
        )
        ax.set_title(f"{model_family} | {lang} | macro-F1 (zero-shot)")
        ax.set_xlabel("Variant")
        ax.set_ylabel("Context")

        file_path = save_dir / f"heatmap__{model_family}__{lang}__zero_shot.png"
        save_plot(fig, file_path)


def plot_performance_heatmap(
    master_df: pd.DataFrame,
    save_dir: Path,
    setting: str = "zero_shot",
    model_family: str = "mBERT",
    include_mwe_segment: bool = True,
    language_setup: str = "isolated",
    train_lang_joint: str = "EN_PT_GL",
    eval_languages: list[str] = ("EN", "PT", "Joint"),
    metric: str = "macro_f1",
) -> None:
    '''
    Heatmap: rows=variant, cols=(language × context_label), values=macro_f1.
    '''
    df = prepare_master_with_language_setup(
        master_df,
        setting=setting,
        model_family=model_family,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=("EN", "PT", "GL", "Joint"),
    )
    df = df[df["language_setup"] == language_setup].copy()
    df = df[df["eval_language"].astype(str).isin(list(eval_languages))].copy()

    if df.empty:
        print(f"[heatmap] No data for model_family={model_family}, language_setup={language_setup}, setting={setting}")
        return

    pivot = df.pivot_table(
        index="variant",
        columns=["eval_language", "context_label"],
        values=metric,
        aggfunc="mean",
    )

    # enforce column order
    new_cols = []
    for lang in eval_languages:
        for ctx in CONTEXT_ORDER:
            if (lang, ctx) in pivot.columns:
                new_cols.append((lang, ctx))
    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(new_cols, names=pivot.columns.names))

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", linewidths=0.5, cbar_kws={"label": "Macro F1"}, ax=ax)
    ax.set_title(f"{model_family} | {language_setup} | {setting}: Variant × Context", fontsize=13)
    ax.set_ylabel("Input Variant")
    ax.set_xlabel("Eval language / Context")

    file_path = save_dir / f"heatmap__{model_family}__{setting}__{language_setup}.png"
    save_plot(fig, file_path)





def table3_joint_minus_isolated_deltas(df_lang_setup: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compare multilingual and isolated training by model family.

    For each model family, this creates a table showing how much performance
    changes when moving from isolated to joint training, separated by evaluation
    language and input setup.
    """
    agg = (
        df_lang_setup.groupby(["model_family", "eval_language", "variant", "context_label", "language_setup"], dropna=False)["macro_f1"]
        .mean()
        .reset_index()
    )

    # wide language_setups
    wide_r = agg.pivot_table(
        index=["model_family", "eval_language", "variant", "context_label"],
        columns="language_setup",
        values="macro_f1",
        aggfunc="mean",
    ).reset_index()

    if "isolated" not in wide_r.columns:
        wide_r["isolated"] = float("nan")
    if "joint" not in wide_r.columns:
        wide_r["joint"] = float("nan")

    wide_r["delta"] = wide_r["joint"] - wide_r["isolated"]

    out: dict[str, pd.DataFrame] = {}

    for mf in sorted(wide_r["model_family"].dropna().unique()):
        sub = wide_r[wide_r["model_family"] == mf].copy()

        piv = sub.pivot_table(
            index=["variant", "eval_language"],
            columns="context_label",
            values="delta",
            aggfunc="first",
        ).reset_index()

        if "Full" not in piv.columns or "Target" not in piv.columns:
            # if missing, still build what exists (but you'll likely want to fix coverage)
            piv["Full"] = piv.get("Full", float("nan"))
            piv["Target"] = piv.get("Target", float("nan"))

        piv["Δ"] = piv["Full"] - piv["Target"]

        wide = piv.pivot_table(
            index="variant",
            columns="eval_language",
            values=["Full", "Target", "Δ"],
            aggfunc="first",
        )

        wide.columns = wide.columns.swaplevel(0, 1)  # (language, metric)

        ordered_cols = []
        for lang in [l for l in ["EN", "PT", "GL"] if l in wide.columns.levels[0]]:
            for met in ["Full", "Target", "Δ"]:
                if (lang, met) in wide.columns:
                    ordered_cols.append((lang, met))
        wide = wide.reindex(columns=pd.MultiIndex.from_tuples(ordered_cols, names=["language", "metric"]))

        out[mf] = wide

    return out


def build_table_joint_vs_isolated(
    master_df: pd.DataFrame,
    model_family: str = "mBERT",
    setting: str = "zero_shot",
    include_mwe_segment: bool = True,
    context: str = "previous_target_next",   # Full baseline
    variant: str = "Standard",               # output of normalize_variant
    train_lang_joint: str = "EN_PT_GL",      # your joint training tag in `language`
    eval_languages: tuple[str, ...] = ("EN", "PT", "GL"),
) -> pd.DataFrame:
    """
    Build a language-wise comparison of isolated and joint training.

    The table reports, for each evaluation language, the average score of a
    model trained separately per language versus jointly across languages,
    together with the difference between both language_setups.
    """
    df = master_df.copy()

    df = df[
        (df["setting"] == setting) &
        (df["model_family"] == model_family) &
        (df["include_mwe_segment"] == include_mwe_segment) &
        (df["context"] == context)
    ].copy()

    df = normalize_variant(df)  # adds df["variant"]
    df = df[df["variant"].astype(str) == variant].copy()

    df = df[df["eval_language"].isin(eval_languages)].copy()
    if df.empty:
        raise ValueError("No rows after filtering. Check context/variant/setting/model_family.")

    # isolated: train language == eval language (per_language)
    iso = df[
        (df["language_mode"] == "per_language") &
        (df["language"] == df["eval_language"])
    ].copy()
    iso["language_setup"] = "isolated"

    # joint: multilingual with your joint training tag
    joint = df[
        (df["language_mode"] == "multilingual") &
        (df["language"] == train_lang_joint)
    ].copy()
    joint["language_setup"] = "joint"

    comp = pd.concat([iso, joint], ignore_index=True)
    if comp.empty:
        raise ValueError(
            "No rows for isolated/joint language_setups. "
            "Check language_mode values and train_lang_joint."
        )

    # average over seeds if multiple
    agg = (
        comp.groupby(["eval_language", "language_setup"], dropna=False)["macro_f1"]
        .mean()
        .reset_index()
        .rename(columns={"eval_language": "language"})
    )

    wide = agg.pivot_table(
        index="language",
        columns="language_setup",
        values="macro_f1",
        aggfunc="mean",
    ).reset_index()

    # ensure columns exist
    for c in ["isolated", "joint"]:
        if c not in wide.columns:
            wide[c] = float("nan")

    wide["delta"] = wide["joint"] - wide["isolated"]

    # ordering
    wide["language"] = pd.Categorical(wide["language"], categories=list(eval_languages), ordered=True)
    wide = wide.sort_values("language").reset_index(drop=True)

    return wide


def plot_joint_vs_isolated_connected(
    table: pd.DataFrame,
    out_path: Path,
    title: str = "mBERT: isolated vs joint (Macro-F1)",
) -> None:
    """
    Plot isolated and joint training side by side for each language.

    The figure shows the score change from isolated to joint training and makes
    it easy to see whether multilingual training helps or hurts per language.
    """
    required = {"language", "isolated", "joint"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Table missing columns: {sorted(missing)}")

    plot_df = table.melt(
        id_vars=["language"],
        value_vars=["isolated", "joint"],
        var_name="language_setup",
        value_name="macro_f1",
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # connect isolated -> joint per language
    for lang in table["language"].astype(str).tolist():
        row = table[table["language"].astype(str) == lang].iloc[0]
        ax.plot(["isolated", "joint"], [row["isolated"], row["joint"]], linewidth=1)

    sns.stripplot(
        data=plot_df,
        x="language_setup",
        y="macro_f1",
        hue="language",
        dodge=True,
        size=7,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Macro-F1")
    ax.legend(title="Eval language", bbox_to_anchor=(1.02, 1), loc="upper left")

    save_plot(fig, out_path)


def plot_language_setup_connected_big_figure(
    master_df: pd.DataFrame,
    save_path: Path,
    setting: str = "zero_shot",
    model_family: str = "mBERT",
    include_mwe_segment: bool = True,
    train_lang_joint: str = "EN_PT_GL",
    eval_languages: tuple[str, ...] = ("EN", "PT", "GL"),
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """
    Create an overview figure of isolated versus joint training across setups.

    The plot summarizes how performance changes across evaluation languages,
    context settings, and input variants, giving a broad view of where joint
    training is beneficial or disadvantageous.
    """
    df = prepare_master_with_language_setup(
        master_df,
        setting=setting,
        model_family=model_family,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=eval_languages,
    )
    df.to_csv(save_path/"master_language_setup_change.csv")

    # Mean over seeds / duplicates
    agg = (
        df.groupby(["eval_language","context_label","variant","language_setup"], dropna=False)[metric]
        .mean()
        .reset_index()
    )

    grid = sns.catplot(
        data=agg,
        kind="point",
        x="language_setup",
        y=metric,
        hue="variant",
        row="eval_language",
        col="context_label",
        order=LANGUAGE_SETUP_ORDER,
        hue_order=VARIANT_ORDER,
        dodge=True,
        markers="o",
        linestyles="-",
        height=3.2,
        aspect=1.2,
    )
    grid.set_axis_labels("", "Macro-F1")
    grid.set_titles("{row_name} | {col_name}")
    grid.fig.suptitle(f"Isolated vs Joint (connected) — {model_family} | {setting}", y=1.02)

    file_path = "language_setup_connected__mBERT__zero_shot.png"

    save_plot(grid.fig, save_path/file_path)
    return agg



