import argparse
from pathlib import Path

from experiments.run_experiments import run_experiments
# from analysis.run_analysis import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "evaluate", "analyse"])
    parser.add_argument("arg1", nargs="?", help="Meaning depends on action")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.action == "train":
        if not args.arg1:
            raise SystemExit("train requires arg1 = experiments_path")
        
        run_experiments(Path(args.arg1), args.overwrite)

    '''
    if args.action == "evaluate":
        if not args.arg1:
            raise SystemExit("evaluation requires arg1 = experiment_id")
        
        run_analysis(Path(args.arg1))

    if args.action == "analyse":
        if args.arg1 is None:
            run_analysis()  
        else:
            run_analysis(Path(args.arg1))
    '''
    
if __name__ == "__main__":
    main()