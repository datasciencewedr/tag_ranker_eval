"""
Main script for mail urgency ranking models pipeline.
This pipeline handles data preprocessing, training and evaluation phases of multiple ranking models, tracking results with MLflow.

Usage: 
python src/main.py --config config/config.json

Project Structure:
    - config/: Configuration files and loaders
    - data_handling/: Data preprocessing modules
    - core/: ML pipeline components (training, evaluation)
    - utils/: Common utilities (logging, paths)
"""

import click
import mlflow

from src.configuration.paths_manager import ProjectPaths
from src.configuration.config_loader import ConfigLoader
from src.data_handling.preprocessor import DataPreprocessor
from src.core.model_trainer import ModelTrainer
from src.core.model_evaluator import ModelEvaluator
from src.utils.logger import setup_logger


@click.command()
@click.option("--config", required=True, help="Path to config file")
def main(config: str) -> None:
    """
    Execute the complete ML pipeline for email priority ranker.

    Args:
        config: Path to the configuration JSON file
            Defines models, hyperparameters, metrics, and paths
    
    Raises:
        FileNotFoundError: If config or data files don't exist
        ValueError: If configuration format is invalid
        RuntimeError: If pipeline execution fails
    """

    logger = setup_logger(__name__)
    config_loader = ConfigLoader(config)
    experiment_config = config_loader.get_experiment_config()
    paths = ProjectPaths(generic_experiment_name=experiment_config["name"])
    
    try:
        logger.info("Starting pipeline with experiment: %s", paths.experiment_name)
        mlflow.set_experiment(paths.experiment_name)
        
        with mlflow.start_run(run_name="complete_pipeline") as parent_run:
            mlflow.set_tag("timestamp", paths.TIMESTAMP)
            
            logger.info("Preparing data...")
            X_train, X_test, y_train, y_test = DataPreprocessor(
                # csv_path=experiment_config["data_path"],
                csv_path="C:/Users/LéaLESBATS/Documents/BP2S/pipeline_model_eval/data/raw/mock_data.csv",
                target_col=experiment_config["target_column"]
            ).prepare_split_data()
            
            logger.info("Starting training phase...")
            trainer = ModelTrainer(
                paths=paths,
                config_loader=config_loader
            )
            trainer.train_and_save_models(X_train, y_train)  
            
            logger.info("Starting evaluation phase...")
            evaluator = ModelEvaluator(
                paths=paths,
                config_loader=config_loader
            )
            evaluator.evaluate_models(X_test, y_test)  

            logger.info("Pipeline completed successfully!")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()