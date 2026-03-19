# src/analysis_submodule/stress_masking.py

from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

from config import PATHS
from utils.helper import read_json, write_json
from models.factory import get_model_runner
from data.data import load_data_splits, build_inputs_for_splits
from evaluation.metrics import compute_metrics, make_predictions
from training import get_model


# ============================================================
# 0) MASKING HELPERS (UNCHANGED - USED BY BOTH ISOLATED + MULTI)
# ============================================================

def mask_first_occurrence(string: str, mwe: str, mask_token: str = "[MASK]") -> str:
    return string.replace(mwe, mask_token, 1)

def mask_last_occurrence(string: str, mwe: str, mask_token: str = "[MASK]") -> str:
    i = string.rfind(mwe)
    if i == -1:
        return string
    return string[:i] + mask_token + string[i + len(mwe):]

def mask_all_occurrences(string: str, mwe: str, mask_token: str = "[MASK]") -> str:
    return string.replace(mwe, mask_token)

def mask_all_occurrences_n_mask(string: str, mwe: str, mask_token: str = "[MASK]") -> str:
    n = max(1, len(str(mwe).split()))
    masked = " ".join([mask_token] * n)
    return string.replace(mwe, masked)

def apply_mask(
    df: pd.DataFrame,
    variant: str,
    text_col: str = "input",
    mwe_col: str = "MWE",
    mask_token: str = "[MASK]",
) -> pd.DataFrame:
    res = df.copy()
    texts = res[text_col].astype(str).tolist()
    mwes = res[mwe_col].astype(str).tolist()

    if variant == "first":
        new_texts = [mask_first_occurrence(s, m, mask_token) for s, m in zip(texts, mwes)]
    elif variant == "last":
        new_texts = [mask_last_occurrence(s, m, mask_token) for s, m in zip(texts, mwes)]
    elif variant == "both":
        new_texts = [mask_all_occurrences(s, m, mask_token) for s, m in zip(texts, mwes)]
    elif variant == "both_n_mask":
        new_texts = [mask_all_occurrences_n_mask(s, m, mask_token) for s, m in zip(texts, mwes)]
    else:
        raise ValueError(f"Unknown variant={variant}")

    res[text_col] = new_texts
    return res


# ============================================================
# 1) ISOLATED (OLD) LOGIC - KEEP AS-IS
#    metrics.json is FLAT and you compute ONE macro_f1
# ============================================================

def stress_test_one_run_isolated(exp_dir: Path) -> Optional[Dict[str, Any]]:
    exp_config_path = exp_dir / "experiment_config.json"
    if not exp_config_path.exists():
        return None

    experiment_config = read_json(exp_config_path)
    metrics_normal = read_json(exp_dir / "metrics.json")

    # only handle flat-schema metrics here
    if "per_language" in metrics_normal:
        return None

    runner = get_model_runner(experiment_config["model_family"])

    # rebuild splits
    train_df, val_df, test_df = load_data_splits(experiment_config, PATHS.data_preprocessed)
    train_data, val_data, test_data = build_inputs_for_splits(train_df, val_df, test_df, experiment_config)

    # load saved model
    model, best_params = get_model(experiment_config, exp_dir, train_data, val_data, runner)

    rows = []
    for variant in ["first", "last", "both", "both_n_mask"]:
        masked_test = apply_mask(test_data, variant=variant, text_col="input", mwe_col="MWE")

        _, test_loader, _ = runner.prepare_features(
            params=best_params,
            config=experiment_config,
            train_df=train_data,
            test_df=masked_test,
        )

        proba = runner.predict_proba(model, test_loader)
        preds = make_predictions(proba)
        metrics_masked = compute_metrics(masked_test["label"], preds)

        rows.append({
            "variant": variant,
            "macro_f1_masked": metrics_masked["macro_f1"],
            "delta_macro_f1": float(metrics_masked["macro_f1"] - float(metrics_normal["macro_f1"])),
            "macro_f1_normal": float(metrics_normal["macro_f1"]),
        })

    return {
        "run_dir": str(exp_dir.name).strip().replace(" ", "").replace(",", "_"),
        "setting": experiment_config.get("setting"),
        "language_mode": experiment_config.get("language_mode"),
        "language": str(experiment_config.get("language")).strip().replace(" ", "").replace(",", "_"),
        "model_family": experiment_config.get("model_family"),
        "seed": experiment_config.get("seed"),
        "normal": metrics_normal,
        "masked_variants": rows,
    }


def run_stress_masking_over_all_runs_isolated(experiments_root: Path, results_root: Path) -> None:
    all_rows = []

    processed_one = False

    for experiment_dir in sorted(experiments_root.iterdir()):
        if processed_one:
            break
        if not experiment_dir.is_dir():
            continue
        if not (experiment_dir / "metrics.json").exists():
            continue

        res = stress_test_one_run_isolated(experiment_dir)
        if res is None:
            continue

        # save per-run artifact
        write_json(experiment_dir / "stress_masking.json", res)

        base = {
            "run_dir": res["run_dir"],
            "setting": res["setting"],
            "language_mode": res["language_mode"],
            "language": res["language"],
            "model_family": res["model_family"],
            "seed": res["seed"],
            "macro_f1_normal": float(res["normal"]["macro_f1"]),
        }

        flat = base.copy()
        for r in res["masked_variants"]:
            v = r["variant"]
            flat[f"macro_f1_{v}"] = float(r["macro_f1_masked"])
            flat[f"delta_macro_f1_{v}"] = float(r["delta_macro_f1"])

        all_rows.append(flat)

        processed_one = True

    df = pd.DataFrame(all_rows)

    if not df.empty:
        df["delta_macro_f1_both_n_minus_both"] = (
            df["delta_macro_f1_both_n_mask"] - df["delta_macro_f1_both"]
        )
        df = df.sort_values("delta_macro_f1_both")

    save_path = results_root / "stress_masking_summary_isolated.csv"
    df.to_csv(save_path, index=False)


# ============================================================
# 2) MULTILINGUAL (NEW) LOGIC
#    metrics.json has per_language + test_data has Language column
#    You compute macro_f1 per language and write 1 row per (run, eval_language)
# ============================================================

def _eval_languages_from_test(df: pd.DataFrame) -> List[str]:
    if "Language" not in df.columns:
        raise ValueError(
            f"[stress_masking] multilingual run requires test_data['Language']. Columns={list(df.columns)}"
        )
    langs = sorted(df["Language"].astype(str).dropna().unique().tolist())
    if not langs:
        raise ValueError("[stress_masking] test_data['Language'] has no values")
    return langs


def _compute_macro_f1_per_language(masked_test: pd.DataFrame, preds: List[int]) -> Dict[str, float]:
    if "Language" not in masked_test.columns:
        raise ValueError(f"[stress_masking] masked_test missing 'Language'. Columns={list(masked_test.columns)}")

    out: Dict[str, float] = {}
    for lang in _eval_languages_from_test(masked_test):
        idx = (masked_test["Language"].astype(str) == lang).to_numpy().nonzero()[0]
        if len(idx) == 0:
            continue
        y_true = masked_test["label"].iloc[idx].tolist()
        y_pred = [preds[i] for i in idx.tolist()]
        m = compute_metrics(y_true, y_pred)
        out[lang] = float(m["macro_f1"])
    return out


def stress_test_one_run_multilingual(exp_dir: Path) -> Optional[Dict[str, Any]]:
    exp_config_path = exp_dir / "experiment_config.json"
    if not exp_config_path.exists():
        return None

    experiment_config = read_json(exp_config_path)
    metrics_normal = read_json(exp_dir / "metrics.json")

    # only handle multilingual schema here
    if "per_language" not in metrics_normal:
        return None

    runner = get_model_runner(experiment_config["model_family"])

    train_df, val_df, test_df = load_data_splits(experiment_config, PATHS.data_preprocessed)
    train_data, val_data, test_data = build_inputs_for_splits(train_df, val_df, test_df, experiment_config)

    langs = _eval_languages_from_test(test_data)

    # ensure normal metrics has these languages
    per_lang = metrics_normal.get("per_language", {})
    missing = [l for l in langs if l not in per_lang]
    if missing:
        raise ValueError(
            f"[stress_masking] per_language missing {missing}. available={sorted(per_lang.keys())}"
        )

    model, best_params = get_model(experiment_config, exp_dir, train_data, val_data, runner)

    rows = []
    for variant in ["first", "last", "both", "both_n_mask"]:
        masked_test = apply_mask(test_data, variant=variant, text_col="input", mwe_col="MWE")

        _, test_loader, _ = runner.prepare_features(
            params=best_params,
            config=experiment_config,
            train_df=train_data,
            test_df=masked_test,
        )

        proba = runner.predict_proba(model, test_loader)
        preds = make_predictions(proba)

        masked_f1_by_lang = _compute_macro_f1_per_language(masked_test, preds)

        for lang in langs:
            macro_f1_masked = float(masked_f1_by_lang[lang])
            macro_f1_normal = float(metrics_normal["per_language"][lang]["macro_f1"])
            rows.append({
                "eval_language": lang,
                "variant": variant,
                "macro_f1_masked": macro_f1_masked,
                "macro_f1_normal": macro_f1_normal,
                "delta_macro_f1": float(macro_f1_masked - macro_f1_normal),
            })

    return {
        "run_dir": str(exp_dir.name).strip().replace(" ", "").replace(",", "_"),
        "setting": experiment_config.get("setting"),
        "language_mode": experiment_config.get("language_mode"),
        "language": str(experiment_config.get("language")).strip().replace(" ", "").replace(",", "_"),
        "model_family": experiment_config.get("model_family"),
        "seed": experiment_config.get("seed"),
        "normal": metrics_normal,
        "masked_variants": rows,  # one row per (eval_language, variant)
    }


def run_stress_masking_over_all_runs_multilingual(experiments_root: Path, results_root: Path) -> None:
    all_rows: List[Dict[str, Any]] = []

    processed_one = False

    for experiment_dir in sorted(experiments_root.iterdir()):
        if processed_one:
            break
        if not experiment_dir.is_dir():
            continue
        if not (experiment_dir / "metrics.json").exists():
            continue

        res = stress_test_one_run_multilingual(experiment_dir)
        if res is None:
            continue

        # save per-run artifact
        write_json(experiment_dir / "stress_masking.json", res)

        # flatten into one row per (run_dir, eval_language)
        by_lang: Dict[str, Dict[str, Any]] = {}

        for r in res["masked_variants"]:
            lang = r["eval_language"]
            by_lang.setdefault(
                lang,
                {
                    "run_dir": res["run_dir"],
                    "setting": res["setting"],
                    "language_mode": res["language_mode"],
                    "language": res["language"],  # training spec
                    "model_family": res["model_family"],
                    "seed": res["seed"],
                    "eval_language": lang,
                    "macro_f1_normal": float(r["macro_f1_normal"]),
                },
            )

            v = r["variant"]
            by_lang[lang][f"macro_f1_{v}"] = float(r["macro_f1_masked"])
            by_lang[lang][f"delta_macro_f1_{v}"] = float(r["delta_macro_f1"])

        # derived column per language-row
        for row in by_lang.values():
            if ("delta_macro_f1_both_n_mask" in row) and ("delta_macro_f1_both" in row):
                row["delta_macro_f1_both_n_minus_both"] = (
                    row["delta_macro_f1_both_n_mask"] - row["delta_macro_f1_both"]
                )

        all_rows.extend(by_lang.values())

        processed_one = True

    df = pd.DataFrame(all_rows)

    if not df.empty and "delta_macro_f1_both" in df.columns:
        df = df.sort_values("delta_macro_f1_both")

    save_path = results_root / "stress_masking_summary_multilingual.csv"
    df.to_csv(save_path, index=False)


# ============================================================
# 3) OPTIONAL: single entrypoint to run both
# ============================================================

def run_stress_masking_all(experiments_root: Path, results_root: Path) -> None:
    run_stress_masking_over_all_runs_isolated(experiments_root, results_root)
    run_stress_masking_over_all_runs_multilingual(experiments_root, results_root)