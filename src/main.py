import argparse
from pathlib import Path

from experiments.run_experiments import run_experiments
from config import PATHS
from analysis.run_analysis import run_analysis
from evaluation.run_evaluation import run_evaluation
from utils.helper import ensure_dirs

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "evaluate", "analyse"])
    parser.add_argument("arg1", nargs="?", type=Path, help="Meaning depends on action")
    parser.add_argument("--overwrite", action="store_true")

    # analyse-specific flags
    parser.add_argument("--split", choices=["train", "dev", "test"], help="Split to analyse")

    args = parser.parse_args()

    ensure_dirs(PATHS)

    if args.action == "train":
        if not args.arg1:
            raise SystemExit("train requires arg1 = experiments_path")
        
        run_experiments(args.arg1, args.overwrite)

    if args.action == "evaluate":
        run_evaluation()
    
    if args.action == "analyse":
        if not args.split:
            parser.error("analyse requires --split (train/dev/test)")

        run_analysis(split_type=args.split, project_paths=PATHS)
    
    
if __name__ == "__main__":
    main()