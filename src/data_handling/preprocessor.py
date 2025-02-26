"""
Data preprocessing mocked module for email ranking pipeline.

Features:
- Email body text vectorization using TF-IDF
- Date feature extraction 
- Train/test split handling
"""

import os
from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """
    Handles data loading, preprocessing and train/test splitting.
    
    Preprocessing steps:
    - TF-IDF vectorization of 'mail_body'
    - Temporal feature extraction from 'date'
    """

    def __init__(self, csv_path: str, target_col: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize preprocessor with data configuration.

        Args:
            csv_path: Path to input CSV file
            target_col: Name of target column
            test_size: Proportion of test set (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.csv_path = csv_path
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = self.build_preprocessor()

    def prepare_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load data, preprocess features and split into train/test sets.

        Returns:
            X_train: Processed training features
            X_test: Processed test features
            y_train: Training target values
            y_test: Test target values
        """
        df = self._load_and_validate_data()
        
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )

        X_train = self.preprocessor.fit_transform(X_train_raw)
        X_test = self.preprocessor.transform(X_test_raw)
        
        return X_train, X_test, y_train, y_test

    def _load_and_validate_data(self) -> pd.DataFrame:
        """
        Load and validate CSV data.

        Returns:
            DataFrame containing raw data

        Raises:
            FileNotFoundError: If CSV file doesn't exist
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file {self.csv_path} not found.")
        df = pd.read_csv(self.csv_path)            
        return df

    @staticmethod
    def _parse_date_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract year, month, day features from date column.

        Args:
            df: Input DataFrame with 'date' column

        Returns:
            DataFrame with extracted temporal features
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        return df[["year", "month", "day"]]

    def build_preprocessor(self) -> ColumnTransformer:
        """
        Create scikit-learn preprocessing pipeline.

        Returns:
            ColumnTransformer combining TF-IDF and date feature extraction
        """
        return ColumnTransformer([
            ("tfidf_mail", TfidfVectorizer(), "mail_body"),
            ("date_feats", FunctionTransformer(self._parse_date_features), ["date"])
        ], remainder="drop")