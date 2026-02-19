import argparse
from pathlib import Path

from experiments.run_experiments import run_experiments
from analysis.run_analysis import run_analysis
from config import PATHS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "evaluate", "analyse"])
    parser.add_argument("arg1", nargs="?", type=Path, help="Meaning depends on action")
    parser.add_argument("--overwrite", action="store_true")

    # analyse-specific flags
    parser.add_argument("--split", choices=["train", "dev", "test"], help="Split to analyse")
    parser.add_argument("--setting", choices=["zero_shot", "one_shot"], help="Setting to analyse")

    args = parser.parse_args()

    if args.action == "train":
        if not args.arg1:
            raise SystemExit("train requires arg1 = experiments_path")
        
        run_experiments(args.arg1, args.overwrite)

    '''
    if args.action == "evaluate":
        if not args.arg1:
            raise SystemExit("evaluation requires arg1 = experiment_id")
        
    #    return
    '''

    if args.action == "analyse":
        if args.split is None or args.setting is None:
            parser.error("analyse requires --split and --setting. Example: analyse --split test --setting zero_shot")
        
        run_analysis(setting=args.setting, split_type=args.split, paths=PATHS)

    
    
if __name__ == "__main__":
    main()