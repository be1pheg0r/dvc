import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from prepare_data import config
from typing import Any
from logger import get_logger

logger = get_logger(__name__)
base_dir = pathlib.Path(__file__).resolve().parent.parent.parent

def split_data(config: Any) -> None:
    """
    :param config: Configuration dictionary containing paths and parameters for splitting.
    """
    load_path = base_dir / config["split_data"]['load_path']
    df = pd.read_csv(load_path)
    logger.info("Data loaded successfully from %s", load_path)

    x = df.drop('target', axis=1)
    y = df['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=config["split_data"]['test_size'],
                                                        random_state=config["split_data"]['random_state'])

    train_df = pd.concat([x_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([x_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    logger.info("Data split into training and testing sets.")

    train_path = base_dir / config["split_data"]['train_path']
    test_path = base_dir / config["split_data"]['test_path']

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Training data saved to %s", train_path)
    logger.info("Testing data saved to %s", test_path)


if __name__ == '__main__':
    logger.info(f"Starting {__file__} script execution...")
    config_path = base_dir / 'config.yaml'
    config = config(config_path)
    split_data(config)
    logger.info(f"Finished {__file__} script execution.")