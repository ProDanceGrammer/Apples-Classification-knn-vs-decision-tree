"""
Dataset splitting utilities for machine learning workflows.

This module provides helper functions for splitting datasets into
training and testing subsets. It abstracts away the common logic of
separating features from the target column, applying stratified splitting 
(by default), and ensuring reproducibility with a fixed random seed.

Functions
---------
split_dataset :
    Split a DataFrame into training and test sets for features and target.
"""

import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split


def split_dataset(
    df: DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 210029807
    ) -> tuple[DataFrame, DataFrame, Series, Series]:
    """
    Split a dataset into training and testing sets.

    This function separates the features (X) and the target variable (y),
    and then splits them into training and test subsets.
    80% training and 20% testing by default.

    Parameters
    ----------
    df : DataFrame
        The full dataset containing both features and the target column.
    target : str
        The name of the target column to predict.
    test_size : float, optional (default=0.2)
        Proportion of the dataset to include in the test split.
    random_state : int, optional (default=210029807)
        Seed used to ensure reproducible splits.

    Returns
    -------
    X_train : DataFrame
        Training features.
    X_test : DataFrame
        Test features.
    y_train : Series
        Training target values.
    y_test : Series
        Test target values.
    """

    # Separate features and target
    X = df.drop(columns=target)
    y = df[target]

    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Check shapes of training and set sets
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    return X_train, X_test, y_train, y_test
