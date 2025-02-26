"""
Model training module that handles:
- Training multiple ML models
- Computing training metrics
- Saving models and results
- MLflow experiment tracking
"""

import mlflow
import pickle
from typing import Any, Dict
import numpy as np

from src.core.base_model_handler import BaseModelHandler  

class ModelTrainer(BaseModelHandler):

    def train_and_save_models(
            self, 
            X_train: np.ndarray, 
            y_train: np.ndarray
            ) -> None:
        """
        Train all configured models and save results.

        Args:
            X_train: Training features
            y_train: Training target values
        """
        models = self.config_loader.get_models()
        all_results: Dict[str, Dict[str, Any]] = {}

        for model_name, model in models.items():
            results = self._train_and_save_single_model(model_name, model, X_train, y_train)
            if results:
                all_results[model_name] = results

        summary_path = self._save_summary(all_results, phase="training")
        mlflow.log_artifact(summary_path)

    def _train_and_save_single_model(
            self, 
            model_name: str, 
            model: Any, 
            X_train: np.ndarray, 
            y_train: np.ndarray
            ) -> Dict[str, Any]:
        """
        Train and evaluate a single model.

        Args:
            model_name: Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training target values

        Returns:
            Dictionary containing training metrics
        """
        self.logger.info(f"Training {model_name}...")
        with mlflow.start_run(run_name=f"{model_name}", nested=True):
            mlflow.set_tag("phase", "training")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)
            self._save_model(model, model_name)
            return self._compute_model_metrics(model_name, y_train, y_pred, phase="train")

    def _save_model(
            self, 
            model: Any, 
            model_name: str
            ) -> None:
        """
        Save trained model to disk and MLflow.

        Args:
            model: Trained model instance
            model_name: Name of the model
        """
        model_path = self.paths.current_models_dir / f"{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(model_path)