import argparse
from pathlib import Path

from config import PATHS
from experiments.run_experiments import run_experiments
from evaluation.run_evaluation import run_evaluation
from analysis.evaluate_subslices import evaluate_subslices
from analysis_submodule.analysis_Michele.main_analysis import run_analysis
from utils.helper import ensure_dirs

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "evaluate", "analyse"])
    parser.add_argument("arg1", nargs="?", type=Path, help="Meaning depends on action")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    ensure_dirs(PATHS)

    if args.action == "train":
        if not args.arg1:
            raise SystemExit("train requires arg1 = experiments_path")
        
        run_experiments(args.arg1, args.overwrite)

    if args.action == "evaluate":
        run_evaluation()
    
    if args.action == "analyse":
        evaluate_subslices(project_paths=PATHS)
        run_analysis(PATHS.runs, PATHS.results)
    
if __name__ == "__main__":
    main()