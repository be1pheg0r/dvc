import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import smogn
import yaml
from typing import Any
from logger import get_logger

logger = get_logger(__name__)

base_dir = pathlib.Path(__file__).resolve().parent.parent.parent

def config(path: pathlib.Path) -> Any:
    """
    :param path: Path to the configuration file.
    :return: Parsed YAML configuration.
    """
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def prepare_data(config: Any) -> pd.DataFrame:
    """
    :param path: Path to the input dataset CSV file.
    """

    df = pd.read_csv(base_dir / config["load_data"]['load_path'])
    logger.info("Data loaded successfully.")

    df.rename(columns={config["featurize"]['target_column']: 'target'}, inplace=True)

    x = df.drop('target', axis=1)
    y = df['target']

    z = np.abs(zscore(df))
    df = df[(z < 3).all(axis=1)]

    df = pd.concat([x, y], axis=1)

    scaler = MinMaxScaler()
    x = df.drop('target', axis=1)
    y = df['target']

    x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

    df = pd.concat([x, y], axis=1)
    logger.info("Data preprocessed successfully.")
    return df


def featurize(df: pd.DataFrame, config: Any) -> None:

    logger.info("Starting data resampling...")
    df_resampled = smogn.smoter(
        data=df,
        y="target"
    )
    logger.info("Data resampling completed.")
    path = base_dir / config["featurize"]['save_path']
    df_resampled.to_csv(path, index=False)
    logger.info(f"Resampled data saved to {path}.")


if __name__ == '__main__':
    logger.info(f"Starting {__file__} script execution...")
    config_path = base_dir / 'config.yaml'
    config_data = config(config_path)
    logger.info("Configuration loaded successfully.")

    logger.info("Preparing data...")
    df = prepare_data(config_data)
    logger.info("Data preparation completed.")
    logger.info("Featurizing data...")
    featurize(df, config_data)
    logger.info("Data featurization completed.")
    logger.info(f"{__file__} script execution finished successfully.")