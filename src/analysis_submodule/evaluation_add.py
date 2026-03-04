"""
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
