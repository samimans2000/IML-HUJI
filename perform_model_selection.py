from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from typing import List, Tuple

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps
    # Gaussian noise and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def get_ridge_lasso_scores(train_x: np.ndarray, train_y: np.ndarray, n_evaluations: int,
                           ridge_lasso_values: List[np.ndarray]) -> List[List[np.ndarray]]:
    """
    Get the train and validation scores for Ridge and Lasso regression for different values of
    the regularization parameter

    Parameters
    ----------
    train_x: ndarray of shape (n_samples, n_features)
        Training data
    train_y: ndarray of shape (n_samples, )
        Training responses
    n_evaluations: int
        Number of evaluations to perform
    ridge_lasso_values: list of ndarray of shape (n_evaluations, )
        Different values of the regularization parameter to evaluate

    Returns
    -------
    scores: list of list of ndarray
        Train and validation scores for Ridge and Lasso regression
    """
    ridge_train_scores, ridge_validation_scores = np.zeros(n_evaluations), np.zeros(n_evaluations)
    lasso_train_scores, lasso_validation_scores = np.zeros(n_evaluations), np.zeros(n_evaluations)

    train_x, train_y = shaffle_values(train_x, train_y)  # for cross validation

    for i in range(n_evaluations):
        ridge_train_scores[i], ridge_validation_scores[i] = cross_validate(RidgeRegression(
            lam=ridge_lasso_values[0][i]), train_x, train_y, mean_square_error, 5)
        lasso_train_scores[i], lasso_validation_scores[i] = cross_validate(Lasso(
            alpha=ridge_lasso_values[1][i]), train_x, train_y, mean_square_error, 5)
    return [[ridge_validation_scores, ridge_train_scores], [lasso_validation_scores,
                                                            lasso_train_scores]]


def plot_cv_different_reg_param_for_ridge_lasso(ridge_lasso_values: List[np.ndarray],
                                                scores: List[List[np.ndarray]]) -> None:
    """
    Plot the train- and validation scores for different values of the regularization parameter
    for Ridge- and Lasso regression
    Parameters
    ----------
    ridge_lasso_values: list of ndarray of shape (n_evaluations, )
        Different values of the regularization parameter to evaluate
    scores: list of ndarray
        Train and validation scores for Ridge and Lasso regression
    Returns
    -------
    None
    """
    ridge_vals, lasso_vals = ridge_lasso_values[0], ridge_lasso_values[1]
    ridge_train_scores, ridge_validation_scores = scores[0][1], scores[0][0]
    lasso_train_scores, lasso_validation_scores = scores[1][1], scores[1][0]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Ridge Regression", "Lasso Regression"))
    fig.add_trace(go.Scatter(
        x=ridge_vals, y=ridge_train_scores, name="Ridge Train Score"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ridge_vals, y=ridge_validation_scores, name="Ridge Validation Score"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=lasso_vals, y=lasso_train_scores, name="Lasso Train Score"), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=lasso_vals, y=lasso_validation_scores, name="Lasso Validation Score"), row=1, col=2)

    fig.update_xaxes(title_text=r"$\text{$\lambda$ parameter}$")
    fig.update_layout(title=dict(x=0.5, text=f"Cross-Validation for Ridge and Lasso Regression"),
                      height=600, width=1000, margin=dict(l=50, r=50, t=50, b=50))
    fig.write_image("Q2_cv_different_reg_param_for_ridge_lasso.png")


def best_ridge_vs_best_lasso_vs_least_squares(scores: List[List[np.ndarray]],
                                              ridge_lasso_values: List[np.ndarray],
                                              train_x: np.ndarray, train_y: np.ndarray,
                                              test_x: np.ndarray, test_y: np.ndarray) -> None:
    """
    Fit the best Ridge and Lasso regression models and compare their test errors to the test error
    of the least squares model

    Parameters
    ----------
    scores: list of ndarray
        Train and validation scores for Ridge and Lasso regression
    ridge_lasso_values: list of ndarray of shape (n_evaluations, )
        Different values of the regularization parameter to evaluate
    train_x: ndarray of shape (n_samples, n_features)
        Training data
    train_y: ndarray of shape (n_samples, )
        Training responses
    test_x: ndarray of shape (n_samples, n_features)
        Testing data
    test_y: ndarray of shape (n_samples, )
        Testing responses

    Returns
    -------
    None
    """
    ridge_lambda = ridge_lasso_values[0][np.argmin(scores[0][0])]
    lasso_lambda = ridge_lasso_values[1][np.argmin(scores[1][0])]
    lasso_predict = Lasso(alpha=lasso_lambda).fit(train_x, train_y).predict(test_x)
    best_ridge_regression = np.round(
        RidgeRegression(lam=ridge_lambda).fit(train_x, train_y).loss(test_x, test_y), 3)
    best_lasso_regression = np.round(mean_square_error(test_y, lasso_predict), 3)
    least_squares_regression = np.round(
        LinearRegression().fit(train_x, train_y).loss(test_x, test_y), 3)

    print(f"""
    The Test Errors Of Each Of The Fitted Models:
    Best Ridge Regression: {best_ridge_regression} with lambda={np.round(ridge_lambda, 5)}
    Best Lasso Regression: {best_lasso_regression} with lambda={np.round(lasso_lambda, 5)}
    Least Squares Regression: {least_squares_regression}""")


def shaffle_values(train_x: np.ndarray, train_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shuffle the training data and responses

    Parameters
    ----------
    train_x: ndarray of shape (n_samples, n_features)
        Training data
    train_y: ndarray of shape (n_samples, )
        Training responses

    Returns
    -------
    train_x: ndarray of shape (n_samples, n_features)
        Shuffled training data
    train_y: ndarray of shape (n_samples, )
        Shuffled training responses
    """
    indices = np.arange(len(train_x))
    np.random.shuffle(indices)
    return train_x[indices], train_y[indices]


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization
    parameter values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    x, y = datasets.load_diabetes(return_X_y=True)
    x, y = pd.DataFrame(x), pd.Series(y)
    train_x, train_y, test_x, test_y = split_train_test(x, y, n_samples / len(x))
    train_x, train_y, test_x, test_y = train_x.values, train_y.values, test_x.values, test_y.values

    # Question 2 - Perform CV for different values of the regularization parameter for
    # Ridge and Lasso regressions
    ridge_lasso_values = [np.linspace(0, 0.5, n_evaluations), np.linspace(0.05, 1.5, n_evaluations)]
    scores = get_ridge_lasso_scores(train_x, train_y, n_evaluations, ridge_lasso_values)
    plot_cv_different_reg_param_for_ridge_lasso(ridge_lasso_values, scores)

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge_vs_best_lasso_vs_least_squares(scores, ridge_lasso_values, train_x, train_y, test_x,
                                              test_y)


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
