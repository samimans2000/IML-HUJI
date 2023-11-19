import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from IMLearn.model_selection import cross_validate
from IMLearn.metrics import misclassification_error


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which
        regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure(
        [decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
         go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                    marker_color="black")],
        layout=go.Layout(xaxis=dict(range=xrange),
                         yaxis=dict(range=yrange),
                         title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[
    Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and
    parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value
        and parameters at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values, weights = [], []

    def callback(**kwargs):
        values.append(kwargs['val'])
        weights.append(kwargs['weights'])

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    """
    Compare the convergence rate of the gradient descent algorithm for different fixed learning
    rates
    Parameters
    ----------
    init: np.ndarray, default=np.array([np.sqrt(2), np.e / 3])
        Initial parameter values
    etas: Tuple[float], default=(1, .1, .01, .001)
        Learning rates to be compared

    Returns
    -------
    None
    """

    for L in [L1, L2]:
        fig_2 = go.Figure()
        for eta in etas:
            callback, values, weights = get_gd_state_recorder_callback()
            gd = GradientDescent(learning_rate=FixedLR(eta),
                                 callback=callback, out_type="best").fit(L(init.copy()), None, None)
            # Plot descent path
            fig_1 = plot_descent_path(module=L, descent_path=np.asarray(weights),
                                      title=f"with eta={eta} and {L.__name__} model")
            fig_1.update_layout(height=800, width=1400)
            fig_1.write_image(f"Q1_gd_descent_path_{eta}_{L.__name__}.png")
            # Plot convergence
            fig_2.add_trace(go.Scatter(x=np.linspace(0, len(values) - 1, len(values)), y=values,
                                       mode="markers+lines", name=f"eta={eta}"))
            # Print lowest loss achieved
            print(f"the lowest loss achieved when minimizing {L.__name__} for eta={eta} is: "
                  f"{np.abs(0 - L(gd).compute_output())}")
        fig_2.update_layout(xaxis_title="Iteration", yaxis_title="Norm value",
                            title=dict(
                                text=f"The Convergence Rate with each eta of {L.__name__} model",
                                x=0.5), height=600, width=1200)
        fig_2.write_image(f"Q3_gd_convergence_{L.__name__}.png")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying
    # learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def plotting_convergence_rate(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Plotting convergence rate of logistic regression over SA heart disease data

    Parameters
    ----------
    X_train: pd.DataFrame
        Design matrix of train set
    y_train: pd.Series
        Responses of training samples

    Returns
    -------
    None

    """
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    model = LogisticRegression(solver=GradientDescent(
        learning_rate=FixedLR(1e-4), max_iter=20000)).fit(X_train, y_train)
    y_prob = model.predict_proba(X_train)
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)

    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="",
                         showlegend=False, marker_size=5,
                         hovertemplate=
                         "<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"),
                         height=800, width=1100))
    fig.write_image("Q8_ROC_curve.png")

    optimal_ROC_alpha = thresholds[np.argmax(tpr - fpr)]
    test_error = misclassification_error(y_train, y_prob >= optimal_ROC_alpha)
    print(f"alpha that achieves the optimal ROC value is: {np.round(optimal_ROC_alpha, 3)} with "
          f"test error of: {np.round(test_error, 3)}")


def fit_regularized_logistic_regression(L: Type[BaseModule], X_train: pd.DataFrame,
                                        y_train: pd.Series, X_test: pd.DataFrame,
                                        y_test: pd.Series):
    """
    Fit regularized logistic regression model with L1 or L2 penalty and plot the misclassification
    error as a function of the regularization parameter lambda
    Parameters
    ----------
    L: Type[BaseModule]
        L1 or L2 penalty
    X_train: pd.DataFrame
        Design matrix of train set
    y_train: pd.Series
        Responses of training samples
    X_test: pd.DataFrame
        Design matrix of test set
    y_test: pd.Series
        Responses of test samples

    Returns
    -------
    None
    """
    L_name = "l1" if L == L1 else "l2"
    lambada = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    train_score, validation_score = [], []
    gd = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)
    for lam in lambada:
        logistic_regression = LogisticRegression(solver=gd, penalty=f"{L_name}", alpha=0.5, lam=lam)
        train_score_lam, validation_score_lam = cross_validate(logistic_regression,
                                                               X_train.to_numpy(),
                                                               y_train.to_numpy(),
                                                               misclassification_error)
        train_score.append(train_score_lam)
        validation_score.append(validation_score_lam)

    optimal_lam = lambada[np.argmin(validation_score)]
    optimal_lam_regression = LogisticRegression(solver=gd, penalty=L_name, alpha=0.5,
                                                lam=optimal_lam).fit(X_train.to_numpy(),
                                                                     y_train.to_numpy())
    y_pred = optimal_lam_regression.predict(X_test.to_numpy())
    test_error = np.round(misclassification_error(y_test.to_numpy(), y_pred), 3)
    print(f"best lambda for {L_name} model is: {optimal_lam} and it's test error is: {test_error}")


def fit_logistic_regression():
    # Load and split SA Heart Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    plotting_convergence_rate(X_train, y_train)

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to
    # specify values of regularization parameter
    fit_regularized_logistic_regression(L1, X_train, y_train, X_test, y_test)
    fit_regularized_logistic_regression(L2, X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
