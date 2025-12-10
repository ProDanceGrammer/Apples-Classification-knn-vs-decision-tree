"""
Histogram visualization utilities.

This module provides helper functions for creating distribution plots
of numerical features. These histograms assist in exploratory data 
analysis (EDA) by showing the shape, spread, skewness, and modality 
of each variable, often accompanied by kernel density estimates (KDE)
for smoother distribution visualization.
"""

import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame


def histogram(df: DataFrame, numerical_cols: list) -> None:
    """
    Plot histograms for numerical features.

    This function visualizes the distribution of each numeric column
    using histograms with an optional KDE curve.
       
    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing the features to visualize.
    numerical_cols : list[str]
        List of numerical column names to plot.
        
    Returns
    -------
        The function creates and displays histogram plots.
    """

    plt.figure(figsize=(15, 12))
    for i, col in enumerate(numerical_cols):
        plt.subplot(4, 4, i + 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()



