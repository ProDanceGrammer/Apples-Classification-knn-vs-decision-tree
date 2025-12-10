"""
Boxplot visualization utilities.

This module provides helper functions for visualizing potential outliers in 
numerical datasets using boxplots. These plots are useful for exploratory 
data analysis (EDA), allowing data scientists to quickly inspect the 
distribution and detect extreme values across multiple features.
"""

import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame


def boxplot_outliers(df: DataFrame, numerical_cols: list[str]) -> None:
    """
    Plot boxplots for multiple numerical features.
   
    This function visualizes potential outliers by creating a grid of boxplots
    for selected numerical columns of a DataFrame.
    
    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing the features to visualize.
    numerical_cols : list[str]
        List of numerical column names to plot.
    
    Returns
    -------
        The function creates and displays boxplot figures but does not return
        any object.

    """

    plt.figure(figsize=(15, 12))
    for i, col in enumerate(numerical_cols):
        plt.subplot(4, 4, i + 1)
        sns.boxplot(y=df[col])
        plt.title(f'Box Plot of {col}')
    plt.tight_layout()
    plt.show()


