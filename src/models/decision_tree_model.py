"""
Decision Tree model training and evaluation module.

This module contains functions for training Decision Tree classifiers,
evaluating their performance on training and test datasets, and visualizing
the resulting tree structures. It supports both default Gini-based models
and configurable entropy-based Decision Trees.

Functions
---------
run_decision_tree :
    Trains a Decision Tree classifier with predefined hyperparameters,
    evaluates it on training and test sets, and visualizes the model.
run_decision_tree_with_entropy :
    Trains a Decision Tree classifier using the entropy criterion and
    user-defined hyperparameters, then evaluates and visualizes the model.
_run_decision_tree_pipeline :
    Internal utility function used to reduce duplication by handling the
    common train-predict-evaluate workflow for Decision Tree models.
"""

from pandas import DataFrame, Series
from sklearn.tree import DecisionTreeClassifier

from models.metrics import evaluation_metrics, plot_decision_tree


def run_decision_tree(
        X_train: DataFrame, 
        X_test: DataFrame,
        y_train: Series,
        y_test: Series,
        max_depth: int,
        min_samples_split: int,
        random_state: int
        ) -> None:
    """
    Train and evaluate a Decision Tree classifier.

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

    Returns
    ----------
    None
        The function does not return a value. All evaluation results are 
        printed to the console.
    """

    # Create and train the Decision Tree model
    tree_model = DecisionTreeClassifier(
        max_depth=max_depth, 
        min_samples_split=min_samples_split, 
        random_state=random_state
        )

    _run_decision_tree_pipeline(tree_model, X_train, X_test, y_train, y_test)



def run_decision_tree_with_entropy(
        X_train: DataFrame, 
        X_test: DataFrame,
        y_train: Series,
        y_test: Series,
        max_depth: int,
        min_samples_split: int,
        random_state: int
        ) -> None:
    """
    Train and evaluate a Decision Tree classifier with entropy criterion.

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

    Returns
    -------
    None
        The function does not return a value. All evaluation results are 
        printed to the console.
    """
 
    # Create and train the Decision Tree model with entropy criterion
    tree_model = DecisionTreeClassifier(
        criterion='entropy', 
        max_depth=max_depth, 
        min_samples_split=min_samples_split, 
        random_state=random_state
        )
    
    _run_decision_tree_pipeline(tree_model, X_train, X_test, y_train, y_test)



def _run_decision_tree_pipeline(
        tree_model: DecisionTreeClassifier,
        X_train: DataFrame,
        X_test: DataFrame,
        y_train: Series,
        y_test: Series
    ) -> None:
    """
    Internal helper function that runs the full training-evaluation pipeline
    for a Decision Tree classifier.

    This function trains the provided model, generates predictions for both
    the training and test sets, prints evaluation metrics, and visualizes
    the resulting decision tree structure.

    Parameters
    ----------
    tree_model : DecisionTreeClassifier
        A scikit-learn Decision Tree model instance to be trained.
    X_train : DataFrame
        Training feature set.
    X_test : DataFrame
        Test feature set.
    y_train : Series
        Ground-truth labels for the training set.
    y_test : Series
        Ground-truth labels for the test set.

    Returns
    -------
    None
        The function produces printed evaluation output and tree plots,
        but does not return any objects. It is intended as an internal
        utility for code reuse within the Decision Tree module.
    """
    # Train
    tree_model.fit(X_train, y_train)

    # Predict and evaluate training set
    y_train_pred = tree_model.predict(X_train)
    set_name = "Training"
    evaluation_metrics(y_train, y_train_pred, set_name)
    plot_decision_tree(tree_model, X_train, set_name)

    # Predict and evaluate test set
    y_test_pred = tree_model.predict(X_test)
    set_name = "Test"
    evaluation_metrics(y_test, y_test_pred, set_name)
    plot_decision_tree(tree_model, X_train, set_name)

