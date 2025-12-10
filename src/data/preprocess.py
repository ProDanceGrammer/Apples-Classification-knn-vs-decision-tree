"""
Data loading and preprocessing utilities.

This module contains helper functions for exploratory data analysis (EDA),
data cleaning, outlier detection, and feature scaling. 

Functions
---------
describe_data(df: DataFrame) -> None
    Provides summary statistics, missing values, duplicates, and data types.
clean_data(df: DataFrame) -> DataFrame
    Drops unnecessary columns (e.g., sample IDs).
detect_outliers_iqr(df: DataFrame) -> None
    Detects and prints outliers based on the Interquartile Range (IQR) method.
standard_scale(df: DataFrame, feature_names: list[str]) -> DataFrame
    Applies Z-score standardization to numerical features.
log_scale(df: DataFrame, numerical_cols: list[str]) -> DataFrame
    Applies logarithmic scaling (log1p) to numerical features.
min_max_scale(df: DataFrame, numerical_cols: list[str]) -> DataFrame
    Applies Min-Max scaling to selected numerical columns.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas import DataFrame

def describe_data(df: DataFrame) -> None:
    """
    Describe dataset

    This function shows first 5 rows of the dataset, prints data dimension,
    summary. After it prints missing values and duplicate rows of the dataset.
    Lastly, it shows data types of features of the dataset.
    
    Parameters
    ----------
        df (DataFrame):
            Input DataFrame containing the features to visualize.
        
    Returns
    ----------
        Nothing to return. This function only prints the descriptive statistics
        in console.
    """

    df .head()

    print("\nOriginal data dimension:\n")
    print(df.shape)

    print("\nSummary:\n")
    print((df).describe())

    missing_values = df.isnull().sum()
    print("\nMissing Values (Original Data):\n")
    print(missing_values)

    # Identify duplicates in the DataFrame
    duplicates = df[df.duplicated()]

    # Print the duplicate rows
    print("\nDuplicates: \n")
    print(duplicates)

    # Identify data types and inconsistencies
    print("\nData Types:")
    print(df.dtypes)


def clean_data(df: DataFrame) -> DataFrame:
    """
    Drops an extra column which represents identification numbers of samples
    
    Parameters
    ----------
    df : DataFrame
        A dataset of the project

    Returns:
    df : DataFrame
        Cleaned dataset
    """
    df_cleaned = df.drop(columns=['A_id'])

    return df_cleaned


def detect_outliers_iqr(df: DataFrame) -> None:
    """
    Prints outliers of the dataset based on interquartlie range.
    
    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing the features to visualize.
        
    Returns
    ----------
        None
    """
    numerical_cols = df.columns

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"\nOutliers for {col}:\n", outliers)


def standard_scale(
        df: DataFrame, 
        feature_names: list[str]
        ) -> DataFrame:
    """
    Scale numerical features using StandardScaler.

    This function applies z-score standardization to the selected feature columns
    of the input DataFrame. It returns both the transformed DataFrame and the 
    fitted StandardScaler object, allowing the scaler to be reused later 
    (e.g., on test data or new incoming samples).

    Args:
        df (DataFrame): 
            A pandas DataFrame containing only the numerical features
            to be standardized.
        feature_names (list[str]): 
            List of feature names (column names) to assign to the scaled DataFrame.

    Returns:
        DataFrame:
            Scaled DataFrame with the same feature names.
    """
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=feature_names)

    return df_scaled


def log_scale(df: DataFrame, numerical_cols: list[str]) -> DataFrame:
    """
    Apply logarithmic scaling (log1p) to numerical columns.

    This transformation is useful for reducing right-skewed distributions,
    stabilizing variance, and improving the performance of models that assume
    approximately normal input features.

    The function applies `np.log1p(x)` to each selected column, which safely
    handles zero and small positive values (unlike plain log).

    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing original features.
    numerical_cols : list[str]
        List of column names to apply log1p scaling to.

    Returns
    -------
    DataFrame
        A copy of the DataFrame where the specified numeric columns
        are transformed using logarithmic scaling.
    """
    df_scaled = df.copy()

    for col in numerical_cols:
        df_scaled[col] = np.log1p(df[col])
    
    return df_scaled


def min_max_scale(df: DataFrame, numerical_cols: list[str]) -> DataFrame:
    """
    Apply Min-Max scaling to selected numerical columns.

    This transformation rescales each feature to the range [0, 1] using:

        X_scaled = (X - X_min) / (X_max - X_min)

    Min-Max scaling is useful when:
    - features have different scales,
    - preserving the original distribution's shape matters,
    - algorithms are sensitive to the magnitude of values
      (e.g., K-Means, Neural Networks, distance-based models).

    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing the original features.
    numerical_cols : list[str]
        List of numerical column names to be scaled.

    Returns
    -------
    DataFrame
        A new DataFrame where the selected numerical columns
        are transformed using Min-Max scaling.
    """
    df_scaled = df.copy()

    min_max_scaler = MinMaxScaler()
    df_scaled[numerical_cols] = min_max_scaler.fit_transform(
        df[numerical_cols]
    )
    return df_scaled
