from pathlib import Path
from typing import Dict, Any, Optional, List, Iterable
import re
import pandas as pd

from config import PATHS
from models.factory import get_model_runner
from data.data import load_data_splits, build_inputs_for_splits
from evaluation.metrics import compute_metrics, make_predictions
from utils.helper import read_json
from training import get_model
from data.ner import apply_ner_batch
from data.glosses import get_glosses


MASK_VARIANTS = ["global_single_mask", "global_n_mask", "targeted_context_mask"]


# ============================================================
# helpers
# ============================================================
def _compile_mwe_pattern(mwe: str) -> re.Pattern:
    """Compile a case-insensitive whole-MWE regex pattern."""
    return re.compile(
        r'(?<!\w)' + re.escape(str(mwe)) + r'(?!\w)',
        re.IGNORECASE,
    )

def replace_mwe_occurrences(
    string: str,
    mwe: str,
    replacement: str,
) -> str:
    """Replace case-insensitive whole-MWE occurrences with the given replacement."""
    pattern = _compile_mwe_pattern(mwe)
    return pattern.sub(replacement, str(string))

def count_occurrences(string: str, mwe: str) -> int:
    """Count case-insensitive whole-MWE occurrences in the input string."""
    pattern = _compile_mwe_pattern(mwe)
    return len(pattern.findall(str(string)))


def add_prev_next_occurrence_counts(
    df: pd.DataFrame,
    previous_col: str = "Previous",
    next_col: str = "Next",
    mwe_col: str = "MWE",
    out_col: str = "mwe_occurrences_prev_next",
) -> pd.DataFrame:
    """Add per-sample MWE occurrence counts for previous and next context only"""
    res = df.copy()
    res[out_col] = [
        count_occurrences(str(prev), str(mwe)) + count_occurrences(str(nxt), str(mwe))
        for prev, nxt, mwe in zip(
            res[previous_col].fillna("").tolist(),
            res[next_col].fillna("").tolist(),
            res[mwe_col].tolist(),
        )
    ]
    return res


# ============================================================
# masking helpers
# ============================================================

def mask_all_occurrences(string: str, mwe: str, mask_token: str = "[MASK]") -> str:
    """Replace all case-insensitive whole-MWE occurrences with a single mask token"""
    return replace_mwe_occurrences(string, mwe, mask_token)


def mask_all_occurrences_n_mask(string: str, mwe: str, mask_token: str = "[MASK]") -> str:
    """Replace all case-insensitive whole-MWE occurrences with one mask token per MWE word"""
    n = max(1, len(str(mwe).split()))
    masked = " ".join([mask_token] * n)
    return replace_mwe_occurrences(string, mwe, masked)


def apply_mask(
    df: pd.DataFrame,
    variant: str,
    text_col: str = "input",
    mwe_col: str = "MWE",
    mask_token: str = "[MASK]",
) -> pd.DataFrame:
    """Apply one global masking variant to the input text column and return a copy"""
    res = df.copy()
    texts = res[text_col].astype(str).tolist()
    mwes = res[mwe_col].astype(str).tolist()

    if variant == "global_single_mask":
        new_texts = [mask_all_occurrences(s, m, mask_token) for s, m in zip(texts, mwes)]
    elif variant == "global_n_mask":
        new_texts = [mask_all_occurrences_n_mask(s, m, mask_token) for s, m in zip(texts, mwes)]
    else:
        raise ValueError(f"Unknown variant={variant}")

    res[text_col] = new_texts
    return res


def apply_transform_to_target_text(
    target: str,
    span_text: str,
    transform: str,
) -> str:
    """
    Apply the input transform to the target sentence after masking/no-masking.
    This preserves the highlight setting in the targeted rebuild path.
    """
    transform = str(transform).strip().lower()

    if transform == "none":
        return target

    if transform == "highlight":
        highlighted = f"<MWE> {span_text} </MWE>"
        return replace_mwe_occurrences(target, span_text, highlighted)

    raise ValueError(f"Unknown transform='{transform}'")


def build_targeted_context_mask_input(
    df: pd.DataFrame,
    config: Dict[str, Any],
    mask_token: str = "[MASK]",
    apply_mask: bool = True,
) -> pd.DataFrame:
    """
    Rebuild input by following the same high-level order as the normal input builder:
      1) replace target MWE (optionally)
      2) apply transform (e.g. highlight)
      3) assemble context
      4) prepend MWE segment if requested
      5) apply NER
      6) append glosses

    When apply_mask=False, this acts as a sanity-check rebuild path and should
    reproduce the original input as closely as possible.
    """
    df = df.copy()

    input_variant = config["input_variant"]
    context = input_variant["context"]
    include_mwe = input_variant["include_mwe_segment"]
    features = input_variant["features"]
    transform = input_variant["transform"]

    texts = []
    languages = []

    for _, row in df.iterrows():
        target = str(row["Target"])
        mwe = str(row["MWE"])
        words = mwe.split()

        masked_mwe = " ".join([mask_token] * max(1, len(words)))
        span_for_target = masked_mwe if apply_mask else mwe
        span_for_prefix = masked_mwe if apply_mask else mwe

        # 1) optionally replace the MWE in the target sentence
        if apply_mask:
            target = replace_mwe_occurrences(target, mwe, masked_mwe)

        # 2) preserve transform behavior (especially highlight)
        target = apply_transform_to_target_text(
            target=target,
            span_text=span_for_target,
            transform=transform,
        )

        # 3) assemble context
        if context == "target":
            parts = [target]
        elif context == "previous_target":
            parts = [row["Previous"], target]
        elif context == "target_next":
            parts = [target, row["Next"]]
        elif context == "previous_target_next":
            parts = [row["Previous"], target, row["Next"]]
        else:
            raise ValueError(
                f"Unknown input_variant.context='{context}'. "
                "Expected one of: target, previous_target, target_next, previous_target_next"
            )

        parts = [str(p) for p in parts if pd.notna(p)]
        text = " [SEP] ".join(parts)

        # 4) prepend MWE segment
        if include_mwe:
            text = f"{span_for_prefix} [SEP] {text}"

        texts.append(text)
        languages.append(row["Language"])

    # 5) NER
    if "ner" in features:
        texts = apply_ner_batch(texts, languages)

    # 6) glosses: keep gloss lookup based on original MWE words,
    # but mask only the gloss-prefix MWE when apply_mask=True
    if "glosses" in features:
        updated_texts = []
        for i, row in df.iterrows():
            mwe = str(row["MWE"])
            words = mwe.split()
            masked_mwe = " ".join([mask_token] * max(1, len(words)))
            gloss_prefix = masked_mwe if apply_mask else mwe

            gloss_parts = [gloss_prefix + "."]
            for word in words:
                gloss_parts.extend(get_glosses(word, row["Language"]))

            gloss_segment = " ".join(gloss_parts)
            updated_texts.append(f"{texts[i]} {gloss_segment} [SEP]")

        texts = updated_texts

    df["input"] = texts
    return df


def build_masked_test_data(
    variant: str,
    test_df: pd.DataFrame,
    test_data: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Build masked test data for one variant"""
    if variant == "targeted_context_mask":
        return build_targeted_context_mask_input(test_df, config)
    return apply_mask(test_data, variant=variant, text_col="input", mwe_col="MWE")

def sanity_check_targeted_rebuild_matches_original(exp_dir: Path) -> Dict[str, Any]:
    """
    Rebuild the targeted input path without masking and compare it against the
    original test inputs from build_inputs_for_splits(...).

    This is the key sanity check for the targeted_context_mask implementation.
    """
    experiment_config = read_json(exp_dir / "experiment_config.json")
    runner = get_model_runner(experiment_config["model_family"])

    train_df, val_df, test_df = load_data_splits(experiment_config, PATHS.data_preprocessed)
    train_data, val_data, test_data = build_inputs_for_splits(train_df, val_df, test_df, experiment_config)

    rebuilt = build_targeted_context_mask_input(
        test_df,
        experiment_config,
        apply_mask=False,
    )

    original_inputs = test_data["input"].astype(str).tolist()
    rebuilt_inputs = rebuilt["input"].astype(str).tolist()

    equal_flags = [a == b for a, b in zip(original_inputs, rebuilt_inputs)]
    n_total = len(equal_flags)
    n_equal = int(sum(equal_flags))
    n_diff = int(n_total - n_equal)

    first_diffs = []
    if n_diff > 0:
        for i, (a, b, ok) in enumerate(zip(original_inputs, rebuilt_inputs, equal_flags)):
            if not ok:
                first_diffs.append(
                    {
                        "row_idx": i,
                        "original_input": a,
                        "rebuilt_input": b,
                    }
                )
            if len(first_diffs) >= 5:
                break

    return {
        "run_dir": exp_dir.name,
        "n_total": n_total,
        "n_equal": n_equal,
        "n_diff": n_diff,
        "match_fraction": float(n_equal / n_total) if n_total > 0 else None,
        "first_diffs": first_diffs,
    }

# ============================================================
# generic helpers
# ============================================================

def _clean_id(x: Any) -> str:
    """Normalize identifiers for stable file and table output"""
    return str(x).strip().replace(" ", "").replace(",", "_")


def _iter_run_dirs(experiments_root: Path) -> Iterable[Path]:
    """Yield experiment directories that contain configs and metrics"""
    for d in sorted(experiments_root.iterdir()):
        if d.is_dir() and (d / "metrics.json").exists() and (d / "experiment_config.json").exists():
            yield d


def _compute_macro_f1_per_language(masked_test: pd.DataFrame, preds: List[int]) -> Dict[str, float]:
    """Compute macro-F1 separately for each language in the masked test set"""
    if "Language" not in masked_test.columns:
        raise ValueError(f"[stress_masking] masked_test missing 'Language'. Columns={list(masked_test.columns)}")

    out = {}
    langs = sorted(masked_test["Language"].astype(str).dropna().unique().tolist())

    for lang in langs:
        idx = (masked_test["Language"].astype(str) == lang).to_numpy().nonzero()[0]
        if len(idx) == 0:
            continue
        y_true = masked_test["label"].iloc[idx].tolist()
        y_pred = [preds[i] for i in idx.tolist()]
        out[lang] = float(compute_metrics(y_true, y_pred)["macro_f1"])

    return out

def compute_group_macro_f1_prev_next(
    df: pd.DataFrame,
    preds: List[int],
    label_col: str = "label",
    group_col: str = "mwe_occurrences_prev_next",
    language: Optional[str] = None,
    language_col: str = "Language",
) -> Dict[str, Any]:
    """Compute grouped macro-F1 for samples with and without MWE occurrences in previous/next"""

    if language is not None:
        lang_mask = df[language_col].astype(str) == str(language)
        df = df.loc[lang_mask].copy()
        lang_idx = lang_mask.to_numpy().nonzero()[0]
        preds = [preds[i] for i in lang_idx.tolist()]

    has_mask = df[group_col] > 0
    no_mask = ~has_mask

    has_idx = has_mask.to_numpy().nonzero()[0]
    no_idx = no_mask.to_numpy().nonzero()[0]

    out = {
        "n_has_mwe_prev_next": int(len(has_idx)),
        "n_no_mwe_prev_next": int(len(no_idx)),
        "macro_f1_has_mwe_prev_next": None,
        "macro_f1_no_mwe_prev_next": None,
    }

    if len(has_idx) > 0:
        y_true_has = df.loc[has_mask, label_col].tolist()
        y_pred_has = [preds[i] for i in has_idx.tolist()]
        out["macro_f1_has_mwe_prev_next"] = float(compute_metrics(y_true_has, y_pred_has)["macro_f1"])

    if len(no_idx) > 0:
        y_true_no = df.loc[no_mask, label_col].tolist()
        y_pred_no = [preds[i] for i in no_idx.tolist()]
        out["macro_f1_no_mwe_prev_next"] = float(compute_metrics(y_true_no, y_pred_no)["macro_f1"])

    return out


def _build_base_result(exp_dir: Path, experiment_config: Dict[str, Any], metrics_normal: Dict[str, Any]) -> Dict[str, Any]:
    """Build shared result metadata for one run"""
    return {
        "run_dir": _clean_id(exp_dir.name),
        "setting": experiment_config.get("setting"),
        "language_mode": experiment_config.get("language_mode"),
        "language": _clean_id(experiment_config.get("language")),
        "model_family": experiment_config.get("model_family"),
        "seed": experiment_config.get("seed"),
        "normal": metrics_normal,
    }


# ============================================================
# flatten helpers
# ============================================================

def _flatten_mask_rows_single_language(res: Dict[str, Any]) -> Dict[str, Any]:
    """Convert single-language masking results into one wide summary row"""
    flat = {
        "run_dir": res["run_dir"],
        "setting": res["setting"],
        "language_mode": res["language_mode"],
        "language": res["language"],
        "model_family": res["model_family"],
        "seed": res["seed"],
        "macro_f1_normal": float(res["normal"]["macro_f1"]),
    }

    for r in res["masked_variants"]:
        v = r["variant"]
        flat[f"macro_f1_{v}"] = float(r["macro_f1_masked"])
        flat[f"delta_macro_f1_{v}"] = float(r["delta_macro_f1"])
        flat[f"n_has_mwe_prev_next_{v}"] = int(r["n_has_mwe_prev_next"])
        flat[f"n_no_mwe_prev_next_{v}"] = int(r["n_no_mwe_prev_next"])
        flat[f"macro_f1_has_mwe_prev_next_{v}"] = r["macro_f1_has_mwe_prev_next"]
        flat[f"macro_f1_no_mwe_prev_next_{v}"] = r["macro_f1_no_mwe_prev_next"]

    if ("delta_macro_f1_global_n_mask" in flat) and ("delta_macro_f1_global_single_mask" in flat):
        flat["delta_macro_f1_global_n_minus_single"] = (
            flat["delta_macro_f1_global_n_mask"] - flat["delta_macro_f1_global_single_mask"]
        )

    return flat


def _flatten_mask_rows_per_language(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert multilingual masking results into one wide row per language"""
    by_lang: Dict[str, Dict[str, Any]] = {}

    for r in res["masked_variants"]:
        lang = r["eval_language"]

        by_lang.setdefault(
            lang,
            {
                "run_dir": res["run_dir"],
                "setting": res["setting"],
                "language_mode": res["language_mode"],
                "language": res["language"],
                "model_family": res["model_family"],
                "seed": res["seed"],
                "eval_language": lang,
                "macro_f1_normal": float(r["macro_f1_normal"]),
            },
        )

        v = r["variant"]
        by_lang[lang][f"macro_f1_{v}"] = float(r["macro_f1_masked"])
        by_lang[lang][f"delta_macro_f1_{v}"] = float(r["delta_macro_f1"])
        by_lang[lang][f"n_has_mwe_prev_next_{v}"] = int(r["n_has_mwe_prev_next"])
        by_lang[lang][f"n_no_mwe_prev_next_{v}"] = int(r["n_no_mwe_prev_next"])
        by_lang[lang][f"macro_f1_has_mwe_prev_next_{v}"] = r["macro_f1_has_mwe_prev_next"]
        by_lang[lang][f"macro_f1_no_mwe_prev_next_{v}"] = r["macro_f1_no_mwe_prev_next"]

    for row in by_lang.values():
        if ("delta_macro_f1_global_n_mask" in row) and ("delta_macro_f1_global_single_mask" in row):
            row["delta_macro_f1_global_n_minus_single"] = (
                row["delta_macro_f1_global_n_mask"] - row["delta_macro_f1_global_single_mask"]
            )

    return list(by_lang.values())


# ============================================================
# evaluation
# ============================================================

def stress_test_one_run_monolingual(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """Evaluate masking variants for one single-language experiment run"""
    experiment_config = read_json(exp_dir / "experiment_config.json")
    metrics_normal = read_json(exp_dir / "metrics.json")

    if "per_language" in metrics_normal:
        return None

    runner = get_model_runner(experiment_config["model_family"])

    train_df, val_df, test_df = load_data_splits(experiment_config, PATHS.data_preprocessed)
    test_df = add_prev_next_occurrence_counts(test_df)

    train_data, val_data, test_data = build_inputs_for_splits(train_df, val_df, test_df, experiment_config)
    model, best_params = get_model(experiment_config, exp_dir, train_data, val_data, runner)

    rows = []

    for variant in MASK_VARIANTS:
        masked_test = build_masked_test_data(variant, test_df, test_data, experiment_config)

        _, test_loader, _ = runner.prepare_features(
            params=best_params,
            config=experiment_config,
            train_df=train_data,
            test_df=masked_test,
        )

        preds = make_predictions(runner.predict_proba(model, test_loader))
        metrics_masked = compute_metrics(masked_test["label"], preds)
        group_metrics = compute_group_macro_f1_prev_next(test_df, preds)

        rows.append(
            {
                "variant": variant,
                "macro_f1_masked": float(metrics_masked["macro_f1"]),
                "delta_macro_f1": float(metrics_masked["macro_f1"] - float(metrics_normal["macro_f1"])),
                "macro_f1_normal": float(metrics_normal["macro_f1"]),
                **group_metrics,
            }
        )

    res = _build_base_result(exp_dir, experiment_config, metrics_normal)
    res["masked_variants"] = rows
    return res


def stress_test_one_run_multilingual(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """Evaluate masking variants for one multilingual experiment run"""
    experiment_config = read_json(exp_dir / "experiment_config.json")
    metrics_normal = read_json(exp_dir / "metrics.json")

    if "per_language" not in metrics_normal:
        return None

    runner = get_model_runner(experiment_config["model_family"])

    train_df, val_df, test_df = load_data_splits(experiment_config, PATHS.data_preprocessed)
    test_df = add_prev_next_occurrence_counts(test_df)

    train_data, val_data, test_data = build_inputs_for_splits(train_df, val_df, test_df, experiment_config)

    langs = sorted(test_data["Language"].astype(str).dropna().unique().tolist())
    per_lang = metrics_normal.get("per_language", {})
    missing = [l for l in langs if l not in per_lang]
    if missing:
        raise ValueError(f"[stress_masking] per_language missing {missing}. available={sorted(per_lang.keys())}")

    model, best_params = get_model(experiment_config, exp_dir, train_data, val_data, runner)

    rows = []

    for variant in MASK_VARIANTS:
        masked_test = build_masked_test_data(variant, test_df, test_data, experiment_config)

        _, test_loader, _ = runner.prepare_features(
            params=best_params,
            config=experiment_config,
            train_df=train_data,
            test_df=masked_test,
        )

        preds = make_predictions(runner.predict_proba(model, test_loader))
        masked_f1_by_lang = _compute_macro_f1_per_language(masked_test, preds)

        for lang in langs:
            macro_f1_masked = float(masked_f1_by_lang[lang])
            macro_f1_normal = float(metrics_normal["per_language"][lang]["macro_f1"])
            group_metrics = compute_group_macro_f1_prev_next(test_df, preds, language=lang)

            rows.append(
                {
                    "eval_language": lang,
                    "variant": variant,
                    "macro_f1_masked": macro_f1_masked,
                    "macro_f1_normal": macro_f1_normal,
                    "delta_macro_f1": float(macro_f1_masked - macro_f1_normal),
                    **group_metrics,
                }
            )

    res = _build_base_result(exp_dir, experiment_config, metrics_normal)
    res["masked_variants"] = rows
    return res


# ============================================================
# csv tables
# ============================================================

def run_stress_masking_over_all_runs_monolingual(
    experiments_root: Path,
    results_root: Path,
) -> None:
    """Run masking stress tests for all single-language runs and save a summary csv"""
    all_rows = []

    for experiment_dir in _iter_run_dirs(experiments_root):
        res = stress_test_one_run_monolingual(experiment_dir)
        if res is None:
            continue

        all_rows.append(_flatten_mask_rows_single_language(res))

    df = pd.DataFrame(all_rows)
    if not df.empty and "delta_macro_f1_global_single_mask" in df.columns:
        df = df.sort_values("delta_macro_f1_global_single_mask")

    df.to_csv(results_root / "stress_masking_summary_monolingual.csv", index=False)


def run_stress_masking_over_all_runs_multilingual(
    experiments_root: Path,
    results_root: Path,
) -> None:
    """Run masking stress tests for all multilingual runs and save a summary csv"""
    all_rows = []

    for experiment_dir in _iter_run_dirs(experiments_root):
        res = stress_test_one_run_multilingual(experiment_dir)
        if res is None:
            continue

        all_rows.extend(_flatten_mask_rows_per_language(res))

    df = pd.DataFrame(all_rows)
    if not df.empty and "delta_macro_f1_global_single_mask" in df.columns:
        df = df.sort_values("delta_macro_f1_global_single_mask")

    df.to_csv(results_root / "stress_masking_summary_multilingual.csv", index=False)


def run_stress_masking_all(
    experiments_root: Path,
    results_root: Path,
) -> None:
    """Run masking stress tests for all monolingual and multilingual experiment runs"""
    run_stress_masking_over_all_runs_monolingual(experiments_root, results_root)
    run_stress_masking_over_all_runs_multilingual(experiments_root, results_root)


def run_targeted_rebuild_sanity_checks(experiments_root: Path) -> None:
    debug_runs = [
        "zero_shot__EN__previous_target_next_True_none_empty__mBERT__seed51",
        "zero_shot__EN__previous_target_next_True_highlight_empty__mBERT__seed51",
    ]

    for run_name in debug_runs:
        exp_dir = experiments_root / run_name
        res = sanity_check_targeted_rebuild_matches_original(exp_dir)

        print("\n" + "=" * 80)
        print(res["run_dir"])
        print(f"n_total={res['n_total']}")
        print(f"n_equal={res['n_equal']}")
        print(f"n_diff={res['n_diff']}")
        print(f"match_fraction={res['match_fraction']}")

        if res["first_diffs"]:
            print("First diffs:")
            for d in res["first_diffs"]:
                print("-" * 40)
                print(f"row_idx: {d['row_idx']}")
                print("ORIGINAL:")
                print(d["original_input"])
                print("REBUILT:")
                print(d["rebuilt_input"])