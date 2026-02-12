import pandas as pd
from typing import Tuple, Dict, Any
from pathlib import Path

from src.utils.helper import read_csv_data


# ! TODO: finish function to build input variants
def apply_input_variant(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Create the model input text for this config and store it in the `Input` column"""
    df = df.copy()
    df['Input'] = df['Target'].astype(str)
    return df


def load_data_splits(config: Dict[str, Any], data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test data for the chosen setting and build the input variant"""
    train_data_path = data_dir / f"train_{config['setting']}.csv"
    val_data_path = data_dir / f"val_{config['setting']}.csv"
    test_data_path = data_dir / f"test_{config['setting']}.csv"

    train_data = read_csv_data(train_data_path)
    val_data = read_csv_data(val_data_path)
    test_data = read_csv_data(test_data_path)

    train_data = train_data[train_data['Language'] == config['language']].copy()
    val_data = val_data[val_data['Language'] == config['language']].copy()
    test_data = test_data[test_data['Language'] == config['language']].copy()

    return train_data, val_data, test_data


def build_inputs_for_splits(
    train_data: pd.DataFrame, 
    val_data: pd.DataFrame, 
    test_data: pd.DataFrame, 
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_data = apply_input_variant(train_data, config)
    val_data = apply_input_variant(val_data, config)
    test_data = apply_input_variant(test_data, config)

    return train_data, val_data, test_data

