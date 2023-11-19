import os
from numpy import ndarray
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

train_zipcode_dummies = []
average_values_dict = {}  # dictionary of the average values database


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    global train_zipcode_dummies
    df = pd.concat([X, y], axis=1)

    # drop irrelevant features
    df = df.drop(["id", "date", "lat", "long", "sqft_living15", "sqft_lot15"], axis=1)

    if y is None:  # test set
        # deal with bad values. NOTE: validate values must be >= 0
        df = df.replace(['NA', 'N/A', None, np.nan], -1)

        # set the datatype accordingly
        df["floors"] = df["floors"].astype(int)

        # set average values of bad test values
        df = set_averaged_data(df)

        # categorize non ordered data
        test_zipcode_dummies = pd.get_dummies(df["zipcode"], prefix='zipcode')
        test_zipcode_dummies = test_zipcode_dummies.reindex(columns=train_zipcode_dummies.columns,
                                                            fill_value=0)

        df = pd.concat([df.drop(["zipcode"], axis=1), test_zipcode_dummies], axis=1)
        return df

    else:  # train set
        # drop rows with any column having null data.
        df = df.dropna().drop_duplicates()
        df = df[~df.isin(['NA', 'N/A', None, np.nan]).any(axis=1)]

        # categorize non ordered data
        train_zipcode_dummies = pd.get_dummies(df["zipcode"], prefix='zipcode')
        df = pd.concat([df.drop(["zipcode"], axis=1), train_zipcode_dummies], axis=1)

        # set the datatype accordingly
        df["floors"] = df["floors"].astype(int)

        df = remove_invalid_data(df)
        collect_average_data(df)

        return df.drop("price", axis=1), df["price"]


def collect_average_data(df: pd.DataFrame):
    """
    collect the average value of each column of the train set
    Parameters
    ----------
    df: pd.DataFrame
        train set frame

    Returns
    -------
    None
    """
    for feature in df.columns:
        average_values_dict[feature] = np.mean(df[feature])


def set_averaged_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    set average data values to test set in case of invalid data
    Parameters
    ----------
    df: pd.DataFrame
        test data frame

    Returns
    -------
    test set framed after the replacement of invalid values
    """
    # drop rows with data <= 0 for each of this columns
    df.loc[df["sqft_above"] <= 0, "sqft_above"] = average_values_dict["sqft_above"]
    df.loc[df["sqft_living"] <= 0, "sqft_living"] = average_values_dict["sqft_living"]
    df.loc[df["floors"] <= 0, "floors"] = average_values_dict["floors"]
    df.loc[df["sqft_lot"] <= 0, "sqft_lot"] = average_values_dict["sqft_lot"]
    df.loc[df["yr_built"] <= 0, "yr_built"] = average_values_dict["yr_built"]

    # drop rows with negative data for each of this columns
    df.loc[df["bedrooms"] < 0, "bedrooms"] = average_values_dict["bedrooms"]
    df.loc[df["bathrooms"] < 0, "bathrooms"] = average_values_dict["bathrooms"]
    df.loc[df["sqft_basement"] < 0, "sqft_basement"] = average_values_dict["sqft_basement"]
    df.loc[df["yr_renovated"] < 0, "yr_renovated"] = average_values_dict["yr_renovated"]

    # drop rows with data invalid for their specific values for each of this columns
    df.loc[~df["view"].isin(range(5)), "view"] = average_values_dict["view"]
    df.loc[~df["condition"].isin(range(1, 6)), "condition"] = average_values_dict["condition"]
    df.loc[~df["grade"].isin(range(1, 14)), "grade"] = average_values_dict["grade"]
    df.loc[~df["waterfront"].isin(range(0, 2)), "waterfront"] = average_values_dict["waterfront"]

    return df


def remove_invalid_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    removes invalid data from the given dataframe. in addition, remove extreme samples
    Parameters
    ----------
    df: pd.DataFrame
        represents the dataframe to remove in valid rows from

    Returns
    -------
    the given dataframe without invalid rows - pd.DataFrame

    """
    # drop rows with data <= 0 for each of this columns
    df = df[(df["sqft_above"] > 0) & (df["sqft_living"] > 0) & (df["floors"] > 0) &
            (df["sqft_lot"] > 0) & (df["yr_built"] > 0)]

    # drop rows with negative data for each of this columns
    df = df[(df["price"] >= 0) & (df["bedrooms"] >= 0) & (df["bathrooms"] >= 0) &
            (df["sqft_basement"] >= 0) & (df["yr_renovated"] >= 0)]

    # drop rows with data invalid for their specific values for each of this columns
    df = df[df["view"].isin(range(5)) & df["condition"].isin(range(1, 6)) &
            df["grade"].isin(range(1, 14)) & df["waterfront"].isin(range(0, 2))]

    if df.shape[0] >= 10:
        for col in ["bedrooms", "sqft_lot"]:
            lower_q = df[col].quantile(0.01)
            upper_q = df[col].quantile(0.98)
            df = df[(df[col] < upper_q) & (df[col] > lower_q)]

    return df


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # Create folder for the plots if it doesn't exist
    output_path = output_path + "/Q3_plots"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for feature in X.columns:
        if not feature.startswith("zipcode"):
            x_feature = X[feature]
            y_feature = y
            cov = np.cov(x_feature, y_feature)[1, 0]
            std_x = np.std(x_feature)
            std_y = np.std(y_feature)
            pearson_correlation = np.round(cov / (std_x * std_y), 3)
            create_plot_feature_evaluation(feature, x_feature, y_feature, pearson_correlation,
                                           output_path)


def create_plot_feature_evaluation(feature: str, x: np.array, y: pd.Series,
                                   pearson_correlation: float, output_path: str) -> NoReturn:
    """
    Helper function for creating scatter plot between each feature and the response.
    Parameters
    ----------
    feature: str
        feature name
    x: array-like of shape
        feature samples column
    y: array-like of shape
        response vector to evaluate against
    pearson_correlation: float
        pearson correlation between feature and response
    output_path: str
        path to folder in which plots are saved

    Returns
    -------
    None
    """
    fig = go.Figure(go.Scatter(x=x, y=y, mode='markers', marker=dict(color="black")),
                    layout=go.Layout(
                        title=dict(
                            text=f"{feature} feature vs. Response<br>Pearson Correlation between "
                                 f"{feature} and response is: {pearson_correlation}".title(),
                            x=0.5),
                        xaxis_title=dict(text=fr"{feature}"),
                        yaxis_title=dict(text=r"$response$"),
                        height=600, width=1300))
    fig.write_image(output_path + f"/{feature}.png")


def create_plot_mean_loss(mean_loss: ndarray, std_loss: ndarray) -> NoReturn:
    """
    creates the plot of the mean loss as a function of p%, as well as a confidence interval of
    mean(loss)±2∗std(loss).
    Parameters
    ----------
    mean_loss: ndarray
        the calculated mean loss array
    std_loss: ndarray
        the calculated standard deviation loss
    Returns
    -------
    None
    """
    upper_bound = mean_loss + 2 * std_loss
    lower_bound = mean_loss - 2 * std_loss
    percentage = np.arange(10, 101)
    fig = go.Figure(go.Scatter(x=percentage, y=mean_loss, mode='markers+lines', marker=dict(
        color="black"), name="Mean Prediction"), layout=go.Layout(
        title=dict(
            x=0.5,
            text="Fit model over increasing percentages of the overall training data".title()),
        xaxis_title=dict(text="percentages"),
        yaxis_title=dict(text="mean loss"),
        height=500, width=1100))
    fig.add_trace(go.Scatter(x=percentage, y=upper_bound, mode="lines", name="mean+2*std",
                             line=dict(color="lightgrey"), fill=None, showlegend=False))
    fig.add_trace(go.Scatter(x=percentage, y=lower_bound, mode="lines", name="mean-2*std",
                             line=dict(color="lightgrey"), fill='tonexty', showlegend=False))
    fig.write_image("Q4_increasing_percentages_of_the_overall_training_data.png")


def fit_model_of_increasing_percentages(train_x: Union[ndarray, pd.DataFrame],
                                        train_y: Union[pd.Series, ndarray],
                                        test_x: Union[pd.DataFrame, ndarray],
                                        test_y: Union[pd.Series, ndarray]) -> NoReturn:
    """
    Fit model over increasing percentages of the overall training data.
    Parameters
    ----------
    train_x: Union[ndarray, pd.DataFrame]
        train samples vector
    train_y: Union[pd.Series, ndarray]
        train responses vector
    test_x: Union[ndarray, pd.DataFrame]
        test samples vector
    test_y: Union[pd.Series, ndarray]
        test responses vector

    Returns
    -------
    None
    """
    n_data = len(train_x)
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    results = []
    for p in range(10, 101):
        p_result = []
        for i in range(10):
            #   1) Sample p% of the overall training data
            sample_x = train_x.sample(int(n_data * (p / 100.0)))
            sample_y = train_y.loc[sample_x.index]
            #   2) Fit linear model (including intercept) over sampled set
            sample_fit = LinearRegression(include_intercept=True).fit(sample_x, sample_y)
            #   3) Test fitted model over test set
            #   4) Store average and variance of loss over test set
            sample_loss = sample_fit.loss(test_x, test_y)
            p_result.append(sample_loss)
        results.append(p_result)
    # Then plot average loss as function of training size with error ribbon of size
    # (mean-2*std, mean+2*std)
    mean_loss = np.mean(results, axis=1)
    std_loss = np.std(results, axis=1)
    create_plot_mean_loss(mean_loss, std_loss)


if __name__ == '__main__':
    np.random.seed(0)

    df = pd.read_csv("../datasets/house_prices.csv")
    df = df[df["price"].notna() & df["price"] >= 0]
    data_frame = df.drop("price", axis=1)
    series = df["price"]

    # Question 1 - split data into train and test sets
    train_x_p, train_y_p, test_x_p, test_y_p = split_train_test(data_frame, series)

    # Question 2 - Preprocessing of housing prices dataset
    train_x, train_y = preprocess_data(train_x_p, train_y_p)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_x, train_y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    test_x = preprocess_data(test_x_p)
    fit_model_of_increasing_percentages(train_x, train_y, test_x, test_y_p)
