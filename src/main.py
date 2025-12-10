from config_loader import load_config
from data.load_data import load_apple_data
from data.preprocess import (
    describe_data, 
    clean_data, 
    detect_outliers_iqr,
    standard_scale, 
    log_scale, 
    min_max_scale
    )
from data.split import split_dataset
from models.knn_model import run_knn
from models.decision_tree_model import run_decision_tree, run_decision_tree_with_entropy
from visualization.boxplots import boxplot_outliers
from visualization.histograms import histogram
from visualization.scatter_plots import scatter_plot


# Load configurations
config = load_config()

# General parameters
SEED = config.get("seed", 42)
TEST_SIZE = config.get("test_size", 0.2)

# Models paramaeters
MODEL_PARAMS = config.get("model", {})
N_NEIGHBORS = MODEL_PARAMS.get("n_neighbors", 9)
MAX_DEPTH = MODEL_PARAMS.get("max_depth", 9)
MIN_SAMPLES_SPLIT = MODEL_PARAMS.get("min_samples_split", 2)


def main():

    # -----------------------------
    # 1. Load, describe, clean data
    # -----------------------------
    df = load_apple_data()
    describe_data(df)
    df = clean_data(df)
    detect_outliers_iqr(df)

    numerical_cols = df.columns

    # -----------------------------
    # 2. Exploratory Visualization
    # -----------------------------
    target = 'Quality'    
    histogram(df, numerical_cols)
    boxplot_outliers(df, numerical_cols)
    scatter_plot(df, numerical_cols, target)


    # -----------------------------
    # 3. Data Normalization
    # -----------------------------

    # Data scaling using Z-Score Standardization
    df_scaled = standard_scale(df, numerical_cols)
    boxplot_outliers(df_scaled, numerical_cols)

    # Data scaling using Log Normalization
    df_scaled = log_scale(df, numerical_cols)
    boxplot_outliers(df_scaled, numerical_cols)

    # Data scaling using Min-Max Scaling
    df_scaled = min_max_scale(df, numerical_cols)
    boxplot_outliers(df_scaled, numerical_cols)


    # -----------------------------
    # 4. Train/Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = split_dataset(df_scaled, target, TEST_SIZE, SEED)

    # -----------------------------
    # 5. Model Training and Evaluation
    # -----------------------------

    # kNN Classification
    run_knn(X_train, X_test, y_train, y_test, N_NEIGHBORS)
    
    # Decision Tree Classification
    run_decision_tree(
        X_train, 
        X_test, 
        y_train, 
        y_test, 
        MAX_DEPTH,
        MIN_SAMPLES_SPLIT,
        SEED
        )

    # Decision Tree Classification model with entropy criterion
    run_decision_tree_with_entropy(
        X_train, 
        X_test, 
        y_train, 
        y_test, 
        MAX_DEPTH,
        MIN_SAMPLES_SPLIT,
        SEED
        )


if __name__ == "__main__":
    main()