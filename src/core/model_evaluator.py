import mlflow
import pickle
from pathlib import Path
from typing import Dict, Any
import numpy as np

from src.core.base_model_handler import BaseModelHandler  

class ModelEvaluator(BaseModelHandler):
    """
    Model evaluation module that handles:
    - Loading trained models
    - Computing test metrics
    - Saving evaluation results
    - MLflow experiment tracking for test phase
    """

    def evaluate_models(
            self, 
            X_test: np.ndarray, 
            y_test: np.ndarray
            ) -> None:
        """
        Evaluate all trained models on test data.

        Args:
            X_test: Test features
            y_test: Test target values
        """        
        models = self.config_loader.get_models()
        all_results: Dict[str, Dict[str, Any]] = {}

        for model_name, _ in models.items():
            model_path: Path = self.paths.current_models_dir / f"{model_name}.pkl"
            results = self._evaluate_single_model(model_name, model_path, X_test, y_test)
            if results:  
                all_results[model_name] = results
                
        summary_path = self._save_summary(all_results, phase="testing")
        mlflow.log_artifact(str(summary_path))

    def _evaluate_single_model(
            self, 
            model_name: str, 
            model_path: Path, 
            X_test: np.ndarray, 
            y_test: np.ndarray
            ) -> Dict[str, float]:
        """
        Load and evaluate a single model.

        Args:
            model_name: Name of the model to evaluate
            model_path: Path to saved model file
            X_test: Test features
            y_test: Test target values

        Returns:
            Dictionary containing test metrics
        """        
        if not model_path.exists():
            self.logger.warning(f"Model {model_name} not found at {model_path}")
            return {}

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        self.logger.info(f"Evaluating {model_name}...")
        with mlflow.start_run(run_name=f"{model_name}", nested=True):
            mlflow.set_tag("phase", "test")
            y_pred = model.predict(X_test)

            return self._compute_model_metrics(model_name, y_test, y_pred, phase="test")
