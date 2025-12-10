"""
Data loading utilities for the Apple Quality dataset.

This module provides helper functions for loading and preprocessing
the apple quality dataset used in machine learning experiments.
It includes functionality for reading raw CSV data and converting
categorical quality labels into numerical form.
"""

import pandas as pd
from pandas import DataFrame
from pathlib import Path

def load_apple_data() -> DataFrame:
    """
    Load and preprocess the Apple Quality dataset.

    This function reads the dataset from disk, converts the 'Quality'
    column from categorical labels ('good', 'bad') into numeric values 
    (1 and 0), and returns the processed DataFrame.

    Returns
    -------
    DataFrame
        The preprocessed Apple Quality dataset.
    """
    csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "apple_quality.csv"

    df = pd.read_csv(csv_path)

    # Convert 'Quality' to numerical
    df['Quality'] = df['Quality'].map({'good': 1, 'bad': 0})


    return df