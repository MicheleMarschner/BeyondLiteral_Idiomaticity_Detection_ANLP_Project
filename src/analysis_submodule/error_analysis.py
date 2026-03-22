# analysis_submodule/error_aggregation_with_slices.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PRED_FILE = "test_predictions.csv"
CONFIG_FILE = "experiment_config.json"
INSTANCE_FILE = "instance_overview.csv"


class ErrorAggregatorWithSlices:
    def __init__(
        self,
        experiments_dir: Path,
        output_dir: Path,
        slice_metadata_path: Path | None = None,
    ) -> None:
        self.experiments_dir = Path(experiments_dir)
        self.output_dir = Path(output_dir)
        self.slice_metadata_path = Path(slice_metadata_path) if slice_metadata_path else None
        self.slice_metadata = self._load_slice_metadata()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_slice_metadata(self) -> pd.DataFrame | None:
        if self.slice_metadata_path is None or not self.slice_metadata_path.exists():
            return None

        df = pd.read_csv(self.slice_metadata_path)
        if "id" not in df.columns:
            raise ValueError(f"{self.slice_metadata_path} must contain an 'id' column.")

        df = df.copy()
        df["id"] = df["id"].astype(str)
        return df

    @staticmethod
    def _pick(config: dict[str, Any], key: str, default: Any = None) -> Any:
        if key in config:
            return config[key]
        input_variant = config.get("input_variant", {})
        if isinstance(input_variant, dict) and key in input_variant:
            return input_variant[key]
        return default

    def _extract_run_metadata(self, config: dict[str, Any], run_dir_name: str) -> dict[str, Any]:
        return {
            "run_dir": run_dir_name,
            "model_family": self._pick(config, "model_family"),
            "setting": self._pick(config, "setting"),
            "language_mode": self._pick(config, "language_mode"),
            "language": self._pick(config, "language"),
            "eval_language": self._pick(config, "eval_language"),
            "seed": self._pick(config, "seed"),
            "context": self._pick(config, "context"),
            "features": self._pick(config, "features"),
            "include_mwe_segment": self._pick(config, "include_mwe_segment"),
            "transform": self._pick(config, "transform"),
        }

    def _load_instance_overview(self, run_dir: Path) -> pd.DataFrame | None:
        path = run_dir / INSTANCE_FILE
        if not path.exists():
            return None

        df = pd.read_csv(path)
        if "id" not in df.columns:
            return None

        df = df.copy()
        df["id"] = df["id"].astype(str)

        candidate_cols = [
            "id",
            "label",
            "MWE",
            "mwe",
            "mwe_text",
            "Target",
            "target",
            "Language",
            "language",
            "eval_language",
            "sentence",
            "previous",
            "next",
            "seen_mwe_type",
            "is_ambiguous_mwe",
            "slice_ambiguous",
            "train_mwe_freq_bin",
        ]
        keep = [col for col in candidate_cols if col in df.columns]
        if "id" not in keep:
            keep.insert(0, "id")

        return df[keep].drop_duplicates()

    @staticmethod
    def _merge_on_available_keys(left: pd.DataFrame, right: pd.DataFrame | None) -> pd.DataFrame:
        if right is None:
            return left

        merge_keys = ["id"]
        if "label" in left.columns and "label" in right.columns:
            merge_keys = ["id", "label"]

        extra_cols = [col for col in right.columns if col not in merge_keys]
        right = right[merge_keys + extra_cols].drop_duplicates(subset=merge_keys)

        return left.merge(right, on=merge_keys, how="left")

    def load_all_predictions(self) -> pd.DataFrame:
        all_frames: list[pd.DataFrame] = []

        pred_files = sorted(self.experiments_dir.rglob(PRED_FILE))
        if not pred_files:
            raise FileNotFoundError(f"No {PRED_FILE} files found below {self.experiments_dir}")

        for pred_file in pred_files:
            run_dir = pred_file.parent
            conf_file = run_dir / CONFIG_FILE
            if not conf_file.exists():
                continue

            preds = pd.read_csv(pred_file)
            required = {"id", "label", "test_pred"}
            missing = required - set(preds.columns)
            if missing:
                raise ValueError(f"{pred_file} is missing columns: {sorted(missing)}")

            preds = preds.copy()
            preds["id"] = preds["id"].astype(str)

            config = self._read_json(conf_file)
            meta = self._extract_run_metadata(config, run_dir.name)
            for key, value in meta.items():
                preds[key] = value

            instance_df = self._load_instance_overview(run_dir)
            preds = self._merge_on_available_keys(preds, instance_df)

            all_frames.append(preds)

        df = pd.concat(all_frames, ignore_index=True)

        if self.slice_metadata is not None:
            df = self._merge_on_available_keys(df, self.slice_metadata)

        df["is_correct"] = (df["test_pred"] == df["label"]).astype(int)
        df["is_wrong"] = 1 - df["is_correct"]
        return df

    @staticmethod
    def _first_present(columns: list[str], df: pd.DataFrame) -> str | None:
        for col in columns:
            if col in df.columns:
                return col
        return None

    def aggregate_by_instance(self, df: pd.DataFrame) -> pd.DataFrame:
        group_cols = ["id"]
        descriptive_candidates = [
            "label",
            "MWE",
            "mwe",
            "mwe_text",
            "Target",
            "target",
            "Language",
            "language",
            "eval_language",
            "sentence",
            "seen_mwe_type",
            "is_ambiguous_mwe",
            "slice_ambiguous",
            "train_mwe_freq_bin",
        ]
        for col in descriptive_candidates:
            if col in df.columns and col not in group_cols:
                group_cols.append(col)

        agg_dict: dict[str, tuple[str, str]] = {
            "n_predictions": ("test_pred", "size"),
            "n_runs": ("run_dir", "nunique"),
            "n_correct": ("is_correct", "sum"),
            "n_wrong": ("is_wrong", "sum"),
            "error_rate": ("is_wrong", "mean"),
            "n_models": ("model_family", "nunique"),
        }
        if "test_proba_literal" in df.columns:
            agg_dict["mean_proba_literal"] = ("test_proba_literal", "mean")

        agg = (
            df.groupby(group_cols, dropna=False)
            .agg(**agg_dict)
            .reset_index()
        )

        pred_div = (
            df.groupby(group_cols, dropna=False)["test_pred"]
            .agg(
                n_unique_predictions="nunique",
                majority_prediction=lambda s: s.value_counts(dropna=False).index[0],
                majority_fraction=lambda s: float(s.value_counts(dropna=False).iloc[0] / len(s)),
            )
            .reset_index()
        )

        agg = agg.merge(pred_div, on=group_cols, how="left")
        agg["always_wrong"] = agg["n_wrong"] == agg["n_predictions"]
        agg["always_correct"] = agg["n_correct"] == agg["n_predictions"]
        agg["sometimes_wrong"] = (agg["n_wrong"] > 0) & (agg["n_correct"] > 0)
        agg["disagreement_rate"] = 1.0 - agg["majority_fraction"]

        return agg.sort_values(
            ["error_rate", "n_wrong", "disagreement_rate"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    def aggregate_by_mwe(self, instance_df: pd.DataFrame) -> pd.DataFrame | None:
        mwe_col = self._first_present(["MWE", "mwe", "mwe_text", "Target", "target"], instance_df)
        if mwe_col is None:
            return None

        group_cols = [mwe_col]
        for col in ["Language", "language", "eval_language", "seen_mwe_type", "is_ambiguous_mwe", "slice_ambiguous"]:
            if col in instance_df.columns and col not in group_cols:
                group_cols.append(col)

        out = (
            instance_df.groupby(group_cols, dropna=False)
            .agg(
                n_instances=("id", "nunique"),
                total_predictions=("n_predictions", "sum"),
                total_correct=("n_correct", "sum"),
                total_wrong=("n_wrong", "sum"),
                mean_instance_error_rate=("error_rate", "mean"),
                median_instance_error_rate=("error_rate", "median"),
                n_instances_always_wrong=("always_wrong", "sum"),
                n_instances_always_correct=("always_correct", "sum"),
                mean_disagreement_rate=("disagreement_rate", "mean"),
            )
            .reset_index()
        )
        out["error_rate"] = out["total_wrong"] / out["total_predictions"]

        return out.sort_values(
            ["error_rate", "total_wrong", "mean_disagreement_rate"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    def build_slice_report(self, df: pd.DataFrame) -> pd.DataFrame | None:
        slice_cols = [
            "slice_ambiguous",
            "seen_mwe_type",
            "train_mwe_freq_bin",
            "is_ambiguous_mwe",
            "Language",
            "language",
            "eval_language",
            "label",
        ]

        reports = []
        for col in slice_cols:
            if col not in df.columns:
                continue

            agg_dict: dict[str, tuple[str, str]] = {
                "accuracy": ("is_correct", "mean"),
                "count": ("is_correct", "count"),
                "n_wrong": ("is_wrong", "sum"),
            }
            if "test_proba_literal" in df.columns:
                agg_dict["mean_proba_literal"] = ("test_proba_literal", "mean")

            stats = (
                df.groupby(col, dropna=False)
                .agg(**agg_dict)
                .reset_index()
                .rename(columns={col: "slice_value"})
            )
            stats["slice_type"] = col
            reports.append(stats)

        if not reports:
            return None

        return pd.concat(reports, ignore_index=True)

    def build_systematic_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        group_cols = ["id", "label"]
        for col in [
            "MWE",
            "mwe",
            "mwe_text",
            "Target",
            "target",
            "Language",
            "language",
            "slice_ambiguous",
            "seen_mwe_type",
            "is_ambiguous_mwe",
            "train_mwe_freq_bin",
        ]:
            if col in df.columns and col not in group_cols:
                group_cols.append(col)

        agg_dict: dict[str, tuple[str, str]] = {
            "correct_ratio": ("is_correct", "mean"),
            "n_predictions": ("is_correct", "size"),
            "n_wrong": ("is_wrong", "sum"),
        }
        if "test_proba_literal" in df.columns:
            agg_dict["mean_proba_literal"] = ("test_proba_literal", "mean")

        out = (
            df.groupby(group_cols, dropna=False)
            .agg(**agg_dict)
            .reset_index()
            .sort_values(["correct_ratio", "n_wrong"], ascending=[True, False])
            .reset_index(drop=True)
        )
        return out

    def build_ambiguity_impact(self, df: pd.DataFrame) -> pd.DataFrame | None:
        ambiguity_col = self._first_present(["is_ambiguous_mwe", "slice_ambiguous"], df)
        if ambiguity_col is None:
            return None

        group_cols = [ambiguity_col]
        if "label" in df.columns:
            group_cols.append("label")

        agg_dict: dict[str, tuple[str, str]] = {
            "accuracy": ("is_correct", "mean"),
            "count": ("id", "count"),
            "n_wrong": ("is_wrong", "sum"),
        }
        if "test_proba_literal" in df.columns:
            agg_dict["mean_proba_literal"] = ("test_proba_literal", "mean")

        out = (
            df.groupby(group_cols, dropna=False)
            .agg(**agg_dict)
            .reset_index()
        )
        return out

    def build_overconfident_errors(self, df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame | None:
        if "test_proba_literal" not in df.columns:
            return None

        out = df[(df["is_correct"] == 0) & (df["test_proba_literal"] >= threshold)].copy()
        return out.sort_values("test_proba_literal", ascending=False).reset_index(drop=True)

    def build_perfect_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        perfect = df.groupby("id").filter(lambda x: x["is_correct"].all()).copy()
        return perfect.drop_duplicates(subset=["id"]).reset_index(drop=True)

    def build_model_divergence(self, df: pd.DataFrame) -> pd.DataFrame | None:
        if "model_family" not in df.columns:
            return None

        index_cols = ["id"]
        for col in ["label", "MWE", "mwe", "mwe_text", "Target", "target", "Language", "language"]:
            if col in df.columns and col not in index_cols:
                index_cols.append(col)

        pivot = (
            df.groupby(index_cols + ["model_family"], dropna=False)["is_correct"]
            .mean()
            .unstack("model_family")
            .reset_index()
        )

        model_cols = [col for col in pivot.columns if col not in index_cols]
        if len(model_cols) < 2:
            return None

        pivot["best_model"] = pivot[model_cols].idxmax(axis=1)
        pivot["worst_model"] = pivot[model_cols].idxmin(axis=1)
        pivot["best_score"] = pivot[model_cols].max(axis=1)
        pivot["worst_score"] = pivot[model_cols].min(axis=1)
        pivot["performance_gap"] = pivot["best_score"] - pivot["worst_score"]

        return pivot.sort_values(
            ["performance_gap", "best_score"],
            ascending=[False, False],
        ).reset_index(drop=True)

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        df = self.load_all_predictions()
        instance_df = self.aggregate_by_instance(df)
        mwe_df = self.aggregate_by_mwe(instance_df)
        slice_df = self.build_slice_report(df)
        systematic_df = self.build_systematic_errors(df)
        ambiguity_df = self.build_ambiguity_impact(df)
        overconf_df = self.build_overconfident_errors(df)
        perfect_df = self.build_perfect_samples(df)
        divergence_df = self.build_model_divergence(df)

        df.to_csv(self.output_dir / "prediction_master.csv", index=False)
        instance_df.to_csv(self.output_dir / "instance_error_summary.csv", index=False)
        systematic_df.to_csv(self.output_dir / "report_systematic_errors_with_slices.csv", index=False)
        perfect_df.to_csv(self.output_dir / "samples_perfect_consistency.csv", index=False)

        instance_df.head(100).to_csv(self.output_dir / "instances_most_wrong.csv", index=False)
        instance_df.sort_values(
            ["error_rate", "n_wrong", "disagreement_rate"],
            ascending=[True, True, True],
        ).head(100).to_csv(self.output_dir / "instances_most_right.csv", index=False)

        if mwe_df is not None:
            mwe_df.to_csv(self.output_dir / "mwe_error_summary.csv", index=False)
            mwe_df.head(100).to_csv(self.output_dir / "mwes_most_wrong.csv", index=False)
            mwe_df.sort_values(
                ["error_rate", "total_wrong", "mean_disagreement_rate"],
                ascending=[True, True, True],
            ).head(100).to_csv(self.output_dir / "mwes_most_right.csv", index=False)

        if slice_df is not None:
            slice_df.to_csv(self.output_dir / "report_by_slices.csv", index=False)

        if ambiguity_df is not None:
            ambiguity_df.to_csv(self.output_dir / "report_ambiguity_impact.csv", index=False)

        if overconf_df is not None:
            overconf_df.to_csv(self.output_dir / "errors_overconfident_wrong.csv", index=False)

        if divergence_df is not None:
            divergence_df.to_csv(self.output_dir / "model_divergence_by_instance.csv", index=False)

            model_cols = [
                col for col in divergence_df.columns
                if col not in {
                    "id",
                    "label",
                    "MWE",
                    "mwe",
                    "mwe_text",
                    "Target",
                    "target",
                    "Language",
                    "language",
                    "best_model",
                    "worst_model",
                    "best_score",
                    "worst_score",
                    "performance_gap",
                }
            ]

            if "mBERT" in model_cols and "logreg_tfidf" in model_cols:
                bert_wins = divergence_df[
                    (divergence_df["mBERT"] >= 0.8) &
                    (divergence_df["logreg_tfidf"] <= 0.2)
                ]
                bert_wins.to_csv(
                    self.output_dir / "divergence_mbert_wins_over_logreg_tfidf.csv",
                    index=False,
                )

                logreg_wins = divergence_df[
                    (divergence_df["logreg_tfidf"] >= 0.8) &
                    (divergence_df["mBERT"] <= 0.2)
                ]
                logreg_wins.to_csv(
                    self.output_dir / "divergence_logreg_tfidf_wins_over_mbert.csv",
                    index=False,
                )

        overview = {
            "n_runs": int(df["run_dir"].nunique()),
            "n_prediction_rows": int(len(df)),
            "n_unique_instances": int(df["id"].nunique()),
            "n_instances_always_wrong": int(instance_df["always_wrong"].sum()),
            "n_instances_always_correct": int(instance_df["always_correct"].sum()),
            "n_instances_sometimes_wrong": int(instance_df["sometimes_wrong"].sum()),
        }
        with open(self.output_dir / "error_analysis_overview.json", "w", encoding="utf-8") as f:
            json.dump(overview, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--slice_metadata_path", type=str, default=None)
    args = parser.parse_args()

    aggregator = ErrorAggregatorWithSlices(
        experiments_dir=Path(args.experiments_dir),
        output_dir=Path(args.output_dir),
        slice_metadata_path=Path(args.slice_metadata_path) if args.slice_metadata_path else None,
    )
    aggregator.run()


if __name__ == "__main__":
    main()