"""
KNN model training and evaluation module.

This module provides a function for training a K-Nearest Neighbors
classifier and evaluating its performance on both training and test datasets.

Functions
---------
run_knn :
    Trains a KNN classifier with a given number of neighbors, 
    performs predictions on train and test sets, and prints evaluation metrics.
"""

from pandas import DataFrame, Series
from sklearn.neighbors import KNeighborsClassifier

from models.metrics import evaluation_metrics

def run_knn(
        X_train: DataFrame, 
        X_test: DataFrame,
        y_train: Series,
        y_test: Series,
        n_neighbors: int
        ) -> None:
    """
    Train and evaluate a K-Nearest Neighbors classifier.

    Parameters
    ----------
    X_train : DataFrame
        Training features.
    X_test : DataFrame
        Test features.
    y_train : Series
        Training labels.
    y_test : Series
        Test labels.
    n_neighbors : int
        Number of neighbors to use for KNN.

    Returns
    ----------
    None
        The function does not return a value. All evaluation results are 
        printed to the console.
    """

    # Create and train KNN model 
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)

    # Predict on training set
    y_train_pred = knn_model.predict(X_train)

    # Evaluate on training set
    evaluation_metrics(y_train, y_train_pred, "Training")

    # Predict on test set
    y_test_pred = knn_model.predict(X_test)

    # Evaluate on test set 
    evaluation_metrics(y_test, y_test_pred, "Test")

