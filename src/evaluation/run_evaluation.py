from src.config import PATHS
from src.evaluation.reporting import load_all_runs
from src.utils.helper import ensure_dir


def create_evaluation_overview() -> None:
    df = load_all_runs(PATHS.runs)

    out_dir = PATHS.results
    ensure_dir(out_dir)

    # Save master long table (single seed -> one row per run per eval_language)
    df.to_csv(out_dir / "master_metrics_long.csv", index=False)

def run_evaluation():
    create_evaluation_overview()

