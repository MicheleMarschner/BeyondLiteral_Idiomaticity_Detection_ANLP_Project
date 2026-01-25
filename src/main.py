import argparse

from src.config import PATHS, ensure_dirs, TrainConfig
from src.utils.helper import set_seeds

def main() -> None:
    ensure_dirs(PATHS)
    set_seeds(TrainConfig.seed)
    
if __name__ == "__main__":
    main()