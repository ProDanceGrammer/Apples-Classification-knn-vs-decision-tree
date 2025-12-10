"""
Scatter plot visualization utilities.

This module provides helper functions for plotting scatter plots between
numerical features and a target variable.
"""
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame


def scatter_plot(df: DataFrame, numerical_cols: list[str], target: str) -> None:
    """
    Plot scatter plot for numerical features and target.
       
    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing the features to visualize.
    numerical_cols : list[str]
        List of numerical column names to plot.
    target : str
        Name of the target variable.
    
    Returns
    -------
        The function creates and displays a scatter plot.
    """

    plt.figure(figsize=(15, 12))
    for i, col in enumerate(numerical_cols):
        if col != target:
            plt.subplot(4, 4, i + 1)
            sns.scatterplot(x=df[col], y=df[target], hue=df[target], palette='coolwarm')
            plt.title(f"{col} vs {target}")
            plt.xlabel(col)
            plt.ylabel(target)

    plt.tight_layout()
    plt.show()




