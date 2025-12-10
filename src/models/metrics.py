"""
Evaluation utilities for machine learning models.

This module provides reusable helper functions for evaluating both
regression and classification models. It includes tools for computing
regression error metrics, plotting confusion matrices, generating
classification reports, and visualizing Decision Trees.

Functions
---------
compute_regression_metrics :
    Calculates common regression metrics (MSE, MAE, RMSE).
plot_confusion_matrix :
    Builds and displays a confusion matrix heatmap for classification tasks.
evaluation_metrics :
    High-level wrapper that prints regression metrics, shows a classification
    report, and visualizes a confusion matrix for predicted labels.
plot_decision_tree :
    Visualizes a trained Decision Tree classifier using scikit-learn's plot_tree.
"""


import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    classification_report, 
    confusion_matrix
    )
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns
import matplotlib.pyplot as plt


def compute_regression_metrics(
        y_true: Series, 
        y_pred: ndarray
        ) -> tuple[float, float, float]:
    """
    Compute basic regression error metrics.

    This helper function calculates three commonly used regression metrics:
    Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean
    Squared Error (RMSE). 

    Parameters
    ----------
    y_true : Series
        True target values.

    y_pred_knn : ndarray
        Predicted values output by the model.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing:
        - MSE  (float): Mean Squared Error
        - MAE  (float): Mean Absolute Error
        - RMSE (float): Root Mean Squared Error
    """

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return mse, mae, rmse


def plot_confusion_matrix(
        y_true: Series, 
        y_pred: ndarray, 
        set_name: str):
    """
    Plot a confusion matrix for evaluating classification performance.

    Parameters
    ----------
    y_true : array-like
        Ground-truth (correct) labels for the classification task.
    y_pred : array-like
        Predicted labels returned by the classifier.
    set_name : str
        Label for the dataset (e.g., 'Train', 'Test').

    Description
    -----------
    This function visualizes a confusion matrix, allowing you to assess
    how well a classification model distinguishes between classes.
    It highlights correct predictions on the diagonal and errors off the
    diagonal, helping identify patterns of misclassification.

    Returns
    -------
    None
        The function displays the confusion matrix using a heatmap
        and does not return any value.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples')
    plt.title(f"Confusion Matrix - ({set_name} Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def evaluation_metrics(
        y_true: Series, 
        y_pred: ndarray, 
        set_name: str,
        ):
    """
    Print classification metrics (classification report + confusion matrix)
    and regression metrics.

    Parameters
    ----------
    y_true : Series
        Ground truth values.
    y_pred : ndarray
        Model predictions.
    set_name : str
        Label for the dataset (e.g., 'Train', 'Test').

    Returns
    ----------
    None
        The function prints evaluation metrics (classification report,
        confusion matrix) to the console and does not return any value.
    """
    mse, mae, rmse = compute_regression_metrics(y_true, y_pred)
    
    print(
        f"""{set_name} Set Evaluation:
    MSE: {mse:.4f}
    MAE: {mae:.4f}
    RMSE: {rmse:.4f}""")

    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Confusion matrix plot
    plot_confusion_matrix(y_true, y_pred, set_name)


def plot_decision_tree(
        tree_model: DecisionTreeClassifier,
        X_train: DataFrame,
        set_name: str
    ) -> None:
    """
    Plot a visual representation of a trained Decision Tree classifier.

    Parameters
    ----------
    tree_model : DecisionTreeClassifier
        A fitted Decision Tree model.
    X_train : DataFrame
        Training dataset used to extract feature names for visualization.
    set_name : str
        Label indicating which dataset the plot refers to
        (e.g., "Training" or "Test").

    Returns
    -------
    None
        Displays a matplotlib plot of the decision tree.
    """
    plt.figure(figsize=(15, 10))
    plot_tree(
        tree_model, 
        filled=True, 
        feature_names=X_train.columns, 
        class_names=['Good', 'Bad'], 
        rounded=True
        )
    plt.title(f"Decision Tree Visualization ({set_name} Set)")
    plt.show()
