# Machine Learning Pipeline for Email Urgency Ranking

A comprehensive machine learning pipeline for ranking emails by urgency level, featuring model training, evaluation and experiment tracking.

## Project Overview

This project implements a complete ML pipeline that:
- Trains multiple ML models for email urgency ranking
- Evaluates models with standard ranking metrics
- Tracks experiments and results using MLflow
- Generates detailed performance reports

## Project Structure

```
src/
├── config/                # Configuration management
│   ├── config_loader.py   # Config and model loading
│   └── paths_manager.py   # Project paths handling
├── pipeline/              # Core ML pipeline
│   ├── training.py        # Model training logic
│   └── evaluation.py      # Model evaluation logic
├── data/                  # Data handling
│   └── preprocessing.py   # Data preprocessing
├── utils/                 # Utilities
│   └── logger.py         # Logging configuration
└── main.py               # Entry point
```

## Installation

1. Create a virtual environment:
```bash
python -m venv .bp2s_analyzer
```

2. Activate the environment:
```bash
# Windows
.bp2s_analyzer\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The pipeline is configured through `config.json`:

```json
{
  "experiment_name": "Mail_Urgency_Ranking",
  "metrics": {
    "common": ["mean_squared_error", "mean_absolute_error", "r2_score", "ndcg_score"],
    "model_specific": {}
  },
  "models": [
    {
      "name": "random_forest",
      "class": "sklearn.ensemble.RandomForestRegressor",
      "params": {
        "random_state": 42
      }
    }
  ]
}
```

## Usage

Run the complete pipeline:
```bash
python src/main.py --config src/config/config.json
```

## Results Structure

Results are organized by timestamp for each run:
```
results/
└── run_YYYYMMDD_HHMMSS/
    ├── models/          # Saved models
    │   └── model.pkl
    ├── metrics/         # Detailed metrics
    │   ├── train_metrics.json
    │   └── test_metrics.json
    └── plots/          # Visualizations
```

MLflow tracks:
- Training and test metrics
- Model parameters
- Model artifacts
- Summary reports

## Development

### Adding New Models

1. Add model configuration to `config.json`
2. Model class must implement scikit-learn's estimator interface

### Adding New Metrics

1. Add metric to common_metrics in `config.json`
2. Metric function must follow sklearn's metric signature