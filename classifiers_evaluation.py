from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple, List
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers.
    File is assumed to be an ndarray of shape (n_samples, 3) where the first 2 columns represent
    features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and
    inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable",
                                                                    "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        # callback for saving each loss value
        def losses_callback(fit: Perceptron, x_, y_):
            losses.append(fit.loss(X, y))

        Perceptron(callback=losses_callback).fit(X, y)

        # Plot figure of loss as function of fitting iteration
        create_plot_loss_over_iteration(losses, n)


def create_plot_loss_over_iteration(losses: List[float], n: str) -> None:
    """
    Creates plot figure of loss as function of fitting iteration.
    Parameters
    ----------
    losses: List[float]
        List of loss values in each iteration of the Perceptron algorithm.
    n: str
        Name of dataset

    Returns
    -------
    None
    """
    fig = go.Figure(go.Scatter(x=[i for i in range(len(losses))], y=losses, mode='lines',
                               marker=dict(color="blue")),
                    layout=go.Layout(title=dict(
                        text=f"{n} dataset perceptron training loss values as a "
                             f"function of the training iterations".title(), x=0.5),
                        xaxis_title=dict(text=r"$\text{Training Iterations}$"),
                        yaxis_title=dict(text=r"$\text{Training Loss}$"),
                        height=600, width=1200))
    fig.update_layout(margin=dict(l=50, r=50, t=50, b=50))
    fig.write_image(f"{n.replace(' ', '_')}_training_loss_over_iteration.png")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        gaussian_fit = GaussianNaiveBayes().fit(X, y)
        gaussian_predict = gaussian_fit.predict(X)
        lda_fit = LDA().fit(X, y)
        lda_predict = lda_fit.predict(X)

        # Plot a figure with two subplots, showing the Gaussian Naive Bayes predictions on the left
        # and LDA predictions on the right. Plot title should specify dataset used and subplot
        # titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy

        # calculate prediction accuracy
        gaussian_predict_accuracy = np.round(accuracy(y, gaussian_predict), 3)
        lda_predict_accuracy = np.round(accuracy(y, lda_predict), 3)

        # Create subplots
        fig = create_subplots_with_no_data(f, gaussian_predict_accuracy, lda_predict_accuracy)

        # Add traces for data-points setting symbols and colors
        add_data_points(X, fig, gaussian_predict, lda_predict, y)

        # Add `X` dots specifying fitted Gaussian's' means
        add_markers_of_center_fitted_gaussian(fig, gaussian_fit, lda_fit)

        # Add ellipses depicting the covariances of the fitted Gaussian's
        add_ellipsis_centered_in_gaussian(fig, gaussian_fit, lda_fit)

        fig.write_image(f"gaussian_vs_lda_classifiers_of_{f.replace('.npy', '')}.png")


def create_subplots_with_no_data(dataset, gaussian_predict_accuracy, lda_predict_accuracy):
    """
    Create subplots with no data, only titles and layout.
    Each sub-plot title contain the classifier name and accuracy.
    Parameters
    ----------
    dataset: str
        Name of dataset
    gaussian_predict_accuracy: float
        Gaussian Naive Bayes accuracy
    lda_predict_accuracy: float
        LDA accuracy

    Returns
    -------
    fig: plotly figure
    """
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.05,
                        subplot_titles=[rf"$\text{{Gaussian Naive Bayes predictions with "
                                        rf"{gaussian_predict_accuracy} accuracy}}$",
                                        rf"$\text{{LDA predictions with "
                                        rf"{lda_predict_accuracy} accuracy}}$"])

    fig.update_layout(title=dict(text=f"Gaussian Naive Bayes Predictions VS LDA Predictions "
                                      f"Of {dataset} Dataset", x=0.5), height=600,
                      width=1200, margin=dict(t=70, b=50, l=50, r=50))
    return fig


def add_data_points(X, fig, gaussian_predict, lda_predict, y):
    """
    Add to the given figure 2D scatter-plot of samples with marker color indicating Gaussian
    Naive Bayes predicted class and marker shape indicating true class.
    Add to the given figure 2D scatter-plot of samples, with marker color indicating LDA predicted
    class and marker shape indicating true class.
    Parameters
    ----------
    X: ndarray of shape (n_samples, 2)
        Samples
    fig: plotly figure
        The subplots figure
    gaussian_predict: ndarray of shape (n_samples,)
        Gaussian Naive Bayes predictions
    lda_predict: ndarray of shape (n_samples,)
        LDA predictions
    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    Returns
    -------
    None
    """
    symbol_classifiers = np.array(["circle", "triangle-up", "diamond"])
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                             marker=dict(color=gaussian_predict, symbol=symbol_classifiers[y],
                                         colorscale=[custom[0], custom[-1]]),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                             marker=dict(color=lda_predict, symbol=symbol_classifiers[y],
                                         colorscale=[custom[0], custom[-1]]),
                             showlegend=False), row=1, col=2)


def add_ellipsis_centered_in_gaussian(fig, gaussian_fit, lda_fit):
    """
    Add to the given figure an ellipsis (colored black) centered in Gaussian centers and shape
    dictated by fitted covariance matrix.
    Parameters
    ----------
    fig: plotly figure
        The subplots figure
    gaussian_fit: GaussianNaiveBayes
        Fitted Gaussian Naive Bayes
    lda_fit: LDA
        Fitted LDA

    Returns
    -------
    None
    """
    for i in range(len(gaussian_fit.classes_)):
        fig.add_trace(get_ellipse(gaussian_fit.mu_[i], np.diag(gaussian_fit.vars_[i])), row=1,
                      col=1)

    for i in range(len(lda_fit.classes_)):
        fig.add_trace(get_ellipse(lda_fit.mu_[i], lda_fit.cov_), row=1, col=2)

    fig.update_layout(showlegend=False)


def add_markers_of_center_fitted_gaussian(fig, gaussian_fit, lda_fit):
    """
    Add to the given figure markers (colored black and shaped as 'x') indicating the center of
    fitted Gaussian's.
    Parameters
    ----------
    fig: plotly figure
        The subplots figure
    gaussian_fit: GaussianNaiveBayes
        Fitted Gaussian Naive Bayes
    lda_fit: LDA
        Fitted LDA

    Returns
    -------
    None
    """
    fig.add_trace(go.Scatter(x=gaussian_fit.mu_[:, 0], y=gaussian_fit.mu_[:, 1],
                             mode="markers", marker=dict(color="black", symbol="x"),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=lda_fit.mu_[:, 0], y=lda_fit.mu_[:, 1],
                             mode="markers", marker=dict(color="black", symbol="x"),
                             showlegend=False), row=1, col=2)


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
