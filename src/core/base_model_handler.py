import numpy as np
import json
import mlflow
from typing import Dict, Any

from src.configuration.paths_manager import ProjectPaths
from src.configuration.config_loader import ConfigLoader
from src.utils.logger import setup_logger

class BaseModelHandler:
    """
    Base class for ML model operations.

    Provides common functionality for training and evaluation modules:
    - Metric computation and logging
    - Results saving and summarization
    - MLflow experiment tracking
    """    
    def __init__(
            self, 
            paths: ProjectPaths, 
            config_loader: ConfigLoader
            ):
        """Initialize handler with paths and configuration.
        
        Args:
            paths: Project paths manager
            config_loader: Configuration loader instance
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.paths = paths
        self.config_loader = config_loader
        self.models_infos = config_loader.get_models()
        self.models_metrics = config_loader.get_all_models_metrics()

    def _compute_single_metric(
            self,
            metric_func,
            metric_name: str,
            model_name: str,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            phase: str
        ) -> Any:
        """Compute a single metric value or None if computation fails, and log it with MLflow."""
        try:
            value = metric_func(y_true, y_pred)
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"{phase}_{metric_name}", value)
            return value
        
        except Exception as e:
            self.logger.warning(f"Error computing {metric_name} for {model_name} during {phase} phase: {str(e)}")
            return None

    def _compute_model_metrics(
            self,
            model_name: str,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            phase: str
        ) -> Dict[str, Any]:
        """Compute all metrics for a model, returning adictionary mapping metric names to their values."""
        results = {}
        model_metrics = self.config_loader.get_metrics(model_name)

        self.logger.info(f"Computing metrics for {model_name} during {phase} phase: {list(model_metrics.keys())}")
        for metric_name, metric_func in model_metrics.items():
            results[metric_name] = self._compute_single_metric(metric_func, metric_name, model_name, y_true, y_pred, phase)
        
        self._save_model_metrics(model_name, results, phase)
        return results

    def _save_model_metrics(
            self, 
            model_name: str, 
            results: Dict[str, Any], 
            phase: str
            ) -> None:
        """Save computed metrics for a model to a JSON file locally."""
        metrics_filename = f"{model_name}_{phase}_metrics.json"
        metrics_path = self.paths.metrics_dir / metrics_filename
        
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=4)
        # mlflow.log_artifact(str(metrics_path))
        
        self.logger.info(f"Saved {phase} metrics for {model_name} at {metrics_path}")

    def _save_summary(
            self, 
            all_results: Dict[str, Dict[str, Any]], 
            phase: str
            ) -> str:
        """Save a summary of results for all models to a JSON file locally."""
        summary_filename = f"{phase}_report.json"
        summary_path = self.paths.current_run_dir / summary_filename

        summary = {
            "timestamp": self.paths.TIMESTAMP,
            f"models_{phase}": list(all_results.keys()),
            "metrics_computed": list(next(iter(self.models_metrics.values())).keys()) if self.models_metrics else [],
            "results": all_results
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        self.logger.info(f"Saved {phase} summary at {summary_path}")
        return str(summary_path)
