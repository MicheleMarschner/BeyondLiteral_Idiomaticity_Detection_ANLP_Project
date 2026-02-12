import argparse
from pathlib import Path

from src.experiments.run_experiments import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "evaluate"])
    parser.add_argument("arg1", nargs="?", help="Meaning depends on action")
    parser.add_argument("arg2", nargs="?", help="Meaning depends on action")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.action == "train":
        if not args.arg1:
            raise SystemExit("train requires arg1 = experiments_path")
        
        run_experiments(Path(args.arg1), args.overwrite)
    
if __name__ == "__main__":
    main()