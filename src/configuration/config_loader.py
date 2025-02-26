"""
Configuration management for ML pipeline.

Provides:
- Model configuration handling
- Metrics configuration and loading
- Experiment settings management
"""

import json
import importlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Callable
import numpy as np

@dataclass
class ModelConfig:
    """Single model configuration container."""
    name: str
    class_path: str
    params: Dict[str, Any]

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary."""
        return cls(
            name=config["name"],
            class_path=config["class"],
            params=config.get("params", {})
        )

    def instantiate(self) -> Any:
        """Create model instance from configuration."""
        parts = self.class_path.split(".")
        module_name = ".".join(parts[:-1])
        cls_name = parts[-1]
        
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        return cls(**self.params)

@dataclass
class MetricsConfig:
    """Metrics configuration container."""
    common_metrics: List[str]
    specific_metrics: Dict[str, List[str]]

    def get_metrics_for_model(self, model_name: str) -> List[str]:
        """Get all metrics paths for a specific model."""
        return self.common_metrics + self.specific_metrics.get(model_name, [])

    def load_metrics(self, model_name: str) -> Dict[str, Callable]:
        """Load metric functions for a model."""
        metrics = {}
        for metric_path in self.get_metrics_for_model(model_name):
            func = self._import_metric(metric_path)
            if metric_path == "sklearn.metrics.ndcg_score":
                metrics[metric_path] = lambda y_true, y_pred, f=func: f(
                    np.array([y_true]), np.array([y_pred])
                )
            else:
                metrics[metric_path] = func
        return metrics

    @staticmethod
    def _import_metric(metric_path: str) -> Callable:
        """Import metric function from path."""
        parts = metric_path.split(".")
        module_name = ".".join(parts[:-1])
        metric_name = parts[-1]
        
        try:
            module = importlib.import_module(module_name)
            metric = getattr(module, metric_name)
            
            if not callable(metric):
                raise TypeError(f"Metric '{metric_path}' is not callable")
                
            return metric
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(f"Failed to import metric '{metric_path}': {e}")

class ConfigLoader:
    """Configuration manager for ML pipeline."""

    def __init__(self, config_path: str):
        """Initialize configuration components."""
        self.config = self._load_json(config_path)

        self.model_configs = [
            ModelConfig.from_dict(m) for m in self.config["models"]
        ]
        self.metrics_config = MetricsConfig(
            common_metrics=self.config["metrics"]["common"],
            specific_metrics=self.config["metrics"].get("model_specific", {})
        )
        
        self.experiment_name = self.config["experiment_name"]
        self.data_config = self.config["data"]

    def _load_json(self, path: str) -> Dict[str, Any]:
        """Load and validate JSON configuration."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path) as f:
            return json.load(f)

    def get_models(self) -> Dict[str, Any]:
        """Get instantiated models dictionary."""
        return {
            model.name: model.instantiate()
            for model in self.model_configs
        }

    def get_metrics(self, model_name: str) -> Dict[str, Callable]:
        """Get metrics for a specific model."""
        return self.metrics_config.load_metrics(model_name)

    def get_experiment_config(self) -> Dict[str, Any]:
        """Get experiment configuration."""
        return {
            "name": self.experiment_name,
            "data_path": self.data_config["raw_data_path"],
            "target_column": self.data_config["target_column"]
        }