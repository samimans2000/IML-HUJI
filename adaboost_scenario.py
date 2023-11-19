import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Any


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def plotting_train_test_errors_num_learners(adaboost_fit, n_learners, test_X, test_y, train_X,
                                            train_y):
    """
    Plot the training- and test errors as a function of the number of fitted learners

    Parameters
    ----------
    adaboost_fit: AdaBoost
        AdaBoost object that was fitted to the data
    n_learners: int
        number of learners
    test_X: np.ndarray of shape (n_samples,n_features)
        test samples
    test_y: np.ndarray of shape (n_samples,)
        test labels
    train_X: np.ndarray of shape (n_samples,n_features)
        train samples
    train_y: np.ndarray of shape (n_samples,)
        train labels

    Returns
    -------
    fig: go.Figure
        Figure object of the plot
    """
    train_error, test_error = np.zeros(n_learners), np.zeros(n_learners)
    for i in range(n_learners):
        train_error[i] = (adaboost_fit.partial_loss(train_X, train_y, i + 1))
    for i in range(n_learners):
        test_error[i] = (adaboost_fit.partial_loss(test_X, test_y, i + 1))

    x_values = np.arange(1, n_learners + 1)
    fig = go.Figure(data=[go.Scatter(x=x_values, y=train_error, name='train error', mode='lines'),
                          go.Scatter(x=x_values, y=test_error, name='test error', mode='lines')])
    fig.update_layout(title=dict(
        x=0.5,
        text='Train and Test Error errors as a function of number learners'.title()),
        xaxis_title='number learners',
        yaxis_title='error value', height=600, width=1000, margin=dict(l=50, r=50, t=50, b=50))
    return fig


def plotting_decision_surfaces(adaboost_fit: AdaBoost, T: List[int], lims: np.ndarray,
                               test_X: np.ndarray, test_y: np.ndarray):
    """
    Plot the decision surfaces of AdaBoost for different number of classifiers

    Parameters
    ----------
    adaboost_fit: AdaBoost
        AdaBoost object that was fitted to the data
    T: List[int]
        List of number of classifiers to plot
    lims: np.ndarray
        The limits of the plot
    test_X: np.ndarray of shape (n_samples,n_features)
        test samples
    test_y: np.ndarray of shape (n_samples,)
        test labels

    Returns
    -------
    fig: go.Figure
        Figure object of the plot
    """
    symbols = ["circle" if value == 1 else "x" for value in test_y]
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Number of classifiers is 5",
                                                        "Number of classifiers is 50",
                                                        "Number of classifiers is 100",
                                                        "Number of classifiers is 250"),
                        horizontal_spacing=0.07, vertical_spacing=.07)
    for i in range(len(T)):
        def predict_function(X):
            return adaboost_fit.partial_predict(X, T[i])

        fig.add_traces([decision_surface(predict_function, lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, symbol=symbols,
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(title=dict(text=rf"$\textbf{{Decision Boundaries Of Models}}$", x=0.5),
                      xaxis_title='test column 0', yaxis_title='test column 1',
                      width=1000, height=1000, margin=dict(l=50, r=50, t=50, b=50)).update_xaxes(
        visible=False).update_yaxes(visible=False)
    return fig


def plotting_decision_surface_of_best_performing_ensemble(adaboost_fit: AdaBoost, n_learners: int,
                                                          test_X: np.ndarray, test_y: np.ndarray,
                                                          lims: np.ndarray):
    """
    Plot the decision surface of the best performing ensemble

    Parameters
    ----------
    adaboost_fit: AdaBoost
        AdaBoost object that was fitted to the data
    n_learners: int
        number of learners
    test_X: np.ndarray of shape (n_samples,n_features)
        test samples
    test_y: np.ndarray of shape (n_samples,)
        test labels
    lims: np.ndarray
        The limits of the plot

    Returns
    -------
    fig: go.Figure
        Figure object of the plot
    """
    min_error, t = get_minimum_test_error(adaboost_fit, n_learners, test_X, test_y)

    def predict_function(X):
        return adaboost_fit.partial_predict(X, t + 1)

    symbols = ["circle" if value == 1 else "x" for value in test_y]
    fig = go.Figure(
        data=[decision_surface(predict_function, lims[0], lims[1], showscale=False),
              go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                         showlegend=False, marker=dict(color=test_y, symbol=symbols,
                                                       colorscale=[custom[0], custom[-1]],
                                                       line=dict(color="black", width=1)))])
    fig.update_layout(title=dict(x=0.5, text=f"Best Performing Ensemble With Size:{t + 1} "
                                             f"& Accuracy: {1 - min_error:.2f}"),
                      xaxis_title='feature 1', yaxis_title='feature 2',
                      height=800, width=800, margin=dict(l=50, r=50, t=50, b=50))
    return fig


def get_minimum_test_error(adaboost_fit: AdaBoost, n_learners: int, test_X: np.ndarray,
                           test_y: np.ndarray) -> Tuple[float, Any]:
    """
    Get the minimum test error and the number of classifiers that achieved it

    Parameters
    ----------
    adaboost_fit: AdaBoost
        AdaBoost object that was fitted to the data
    n_learners: int
        The number of classifiers
    test_X: np.ndarray of shape (n_samples,n_features)
        test samples
    test_y: np.ndarray of shape (n_samples,)
        test labels

    Returns
    -------
    Tuple[float, Any]
        The minimum test error and the number of classifiers that achieved it
    """
    test_error = [adaboost_fit.partial_loss(test_X, test_y, i + 1) for i in range(n_learners)]
    min_index = np.argmin(test_error)
    return test_error[min_index], min_index


def plotting_decision_surface_with_weighted_samples(adaboost_fit: AdaBoost, train_X: np.ndarray,
                                                    train_y: np.ndarray, lims: np.ndarray):
    """
    Plot the decision surface with weighted samples

    Parameters
    ----------
    adaboost_fit: AdaBoost
        AdaBoost object that was fitted to the data
    train_X: np.ndarray of shape (n_samples,n_features)
        train samples
    train_y: np.ndarray of shape (n_samples,)
        train labels
    lims: np.ndarray
        The limits of the plot

    Returns
    -------
    fig : go.Figure
        Figure object of the plot
    """
    D = (adaboost_fit.D_ / adaboost_fit.D_.max()) * 15
    symbols = ["circle" if value == 1 else "x" for value in train_y]
    fig = go.Figure(
        data=[decision_surface(adaboost_fit.predict, lims[0], lims[1], showscale=False),
              go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                         showlegend=False, marker=dict(size=D, color=train_y, symbol=symbols,
                                                       colorscale=[custom[0], custom[-1]],
                                                       line=dict(color="black", width=1)))])
    fig.update_layout(title=dict(x=0.5, text=f"Decision Surface With Weighted Samples"),
                      xaxis_title='feature 1', yaxis_title='feature 2',
                      height=800, width=800, margin=dict(l=100, r=100, t=100, b=100))
    return fig


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(
        test_size, noise)
    adaboost_fit = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    noise_tag = "noisy" if noise > 0.0 else "noiseless"

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    fig = plotting_train_test_errors_num_learners(adaboost_fit, n_learners, test_X, test_y, train_X,
                                                  train_y)
    fig.write_image(f"Q1_train_test_error_of_adaBoost_{noise_tag}.png")

    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    # Question 2: Plotting decision surfaces
    fig = plotting_decision_surfaces(adaboost_fit, T, lims, test_X, test_y)
    fig.write_image(f"Q2_decision_surfaces_{noise_tag}.png")

    # Question 3: Decision surface of best performing ensemble
    fig = plotting_decision_surface_of_best_performing_ensemble(adaboost_fit, n_learners, test_X,
                                                                test_y, lims)
    fig.write_image(f"Q3_decision_surface_best_performing_ensemble_{noise_tag}.png")

    # Question 4: Decision surface with weighted samples
    fig = plotting_decision_surface_with_weighted_samples(adaboost_fit, train_X, train_y, lims)
    fig.write_image(f"Q4_decision_surface_with_weighted_samples_{noise_tag}.png")


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0.0)
    fit_and_evaluate_adaboost(noise=0.4)
