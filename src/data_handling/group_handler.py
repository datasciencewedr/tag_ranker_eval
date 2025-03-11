import pandas as pd
import numpy as np

class GroupHandler:
    """Handles creation of groups for Learning to Rank models."""
    
    @staticmethod
    def create_user_day_groups(
        X: pd.DataFrame,
        desk_column: str = 'DESK_ID',
        timestamp_column: str = 'EMAIL_CREATION_DATE'
    ) -> np.ndarray:
        """
        Create groups based on desk and day combination.
        Each group contains all emails from one desk for one day.

        Args:
            X: DataFrame containing email data
            user_column: Name of the user ID column
            timestamp_column: Name of the timestamp column

        Returns:
            List of group sizes for LTR models
        """
        X = X.sort_values([desk_column, timestamp_column])
        X['day'] = X[timestamp_column].dt.date
        
        groups = (X.groupby([desk_column, 'day'])
                  .size()
                  .sort_index()
                  .values)
        
        return groups.tolist()
    


    ## modifier dans model trainer l'appel des groupes pour les modèles types learning to rank