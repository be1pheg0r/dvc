from typing import Any
from prepare_data import config
from datetime import datetime
import joblib
import os
import pathlib
import importlib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from logger import get_logger

logger = get_logger(__name__)
base_dir = pathlib.Path(__file__).resolve().parent.parent.parent


def train_model(config: Any) -> None:
    """
    Обучает модель линейной регрессии на предобработанных данных.
    :return: Обученная модель.
    """
    train_path = base_dir / config["train"]['train_path']
    test_path = base_dir / config["train"]['test_path']

    train, test = pd.read_csv(train_path), pd.read_csv(test_path)

    X_train = train.drop(columns='target')
    y_train = train['target']

    X_val = test.drop(columns='target')
    y_val = test['target']

    model_path = config["train"]["model_path"]

    def load_model(model_path: pathlib.Path) -> Any:
        string_model = str(model_path)
        if not os.path.exists(model_path):
            try:
                module_path = string_model.split('.')[:-1]
                class_name = string_model.split('.')[-1]
                module = importlib.import_module('.'.join(module_path))
                model_class = getattr(module, class_name)
                model = model_class()
                logger.info(f"Model {model_class.__name__} loaded successfully.")
                return model
            except ImportError:
                raise ImportError(f"Model {model_path} is not available. Please check your configuration.")
        else:
            model_path = base_dir / model_path
            logger.info(f"Loading model from {model_path}...")
            return joblib.load(model_path)
    try:
        load_model(model_path)
    except ImportError as e:
        logger.error(f"Error loading model: {e}")
        raise
    params = {
        i: config["train"]["params"][i] for i in config["train"]["params"]
    }
    valid_metrics = {}
    for name in config["train"]["val_metrics"]:
        module = importlib.import_module("sklearn.metrics")
        if hasattr(module, name):
            valid_metrics[name] = getattr(module, name)
        else:
            logger.error(f"Metric {name} not found")
    val_metrics = valid_metrics

    mlflow.set_experiment(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_experiment")
    with mlflow.start_run():
        logger.info("Starting model training...")
        model = load_model(model_path)
        logger.info(f"Model loaded: {model}\n"
                    f"Parameters: {params}")
        grid_search = GridSearchCV(model, params, cv=5, verbose=2, error_score='raise')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_val)
        result_metrics = {}
        for metric_name, metric_func in val_metrics.items():
            result_metrics[metric_name] = metric_func(y_val, y_pred)
        logger.info(f"Validation metrics: {result_metrics}")
        # Логирование параметров и метрик
        for param_name, param_value in best_model.get_params().items():
            mlflow.log_param(param_name, param_value)

        # Логирование модели
        mlflow.sklearn.log_model(best_model, "model")

        logger.info(f"Best parameters: {grid_search.best_params_}")

        save_path = base_dir / config["train"]["save_path"]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        joblib.dump(best_model, save_path / 'best_model.pkl')

    return best_model

if __name__ == '__main__':
    logger.info(f"Starting {__file__} script execution...")
    config_path = base_dir / 'config.yaml'
    config_data = config(config_path)
    train_model(config_data)
    logger.info(f"Finished {__file__} script execution.")