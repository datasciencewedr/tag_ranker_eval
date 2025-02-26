from pathlib import Path 
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ProjectPaths:
    generic_experiment_name: str
    ROOT: Path = Path(__file__).parent.parent.parent
    TIMESTAMP: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @property
    def experiment_name(self) -> str:
        """MLflow experiment name"""
        return f"{self.generic_experiment_name}_{self.TIMESTAMP}"
    
    @property
    def current_run_dir(self) -> Path:
        """Folder name of current run"""
        run_dir = self.ROOT / "results" / f"{self.generic_experiment_name}_run_{self.TIMESTAMP}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    @property
    def current_models_dir(self) -> Path:
        """Folder name for models of current run"""
        models_dir = self.current_run_dir / "models" 
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir
    
    @property
    def metrics_dir(self) -> Path:
        """Folder name for metrics of current run"""
        metrics_dir = self.current_run_dir / "metrics" 
        metrics_dir.mkdir(parents=True, exist_ok=True)
        return metrics_dir
