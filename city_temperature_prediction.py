import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    df = df.dropna()
    df["DayOfYear"] = df["Date"].dt.dayofyear
    # drop rows with data invalid for their specific values for each of this columns
    df = df[df["Year"] >= 0 & df["Month"].isin(range(1, 13)) & df["Day"].isin(range(1, 32))]
    df = df[df["Temp"] >= -20]
    # set columns value to string for use of color discrete in plots
    df["Year"] = df["Year"].astype(str)
    df["Country"] = df["Country"].astype(str)
    return df


def israel_exploring_data(temp: pd.DataFrame) -> pd.DataFrame:
    """
    creates the plots
     - Average daily temperature change as a function of the `DayOfYear`
     - Standard Deviation of Daily Temperatures by Month
    Parameters
    ----------
    temp: pd.DataFrame
            main temperature dataframe

    Returns
    -------
    main temperature dataframe only for israel

    """
    israel_temp = temp[temp["Country"] == "Israel"]
    fig = px.scatter(israel_temp, x="DayOfYear", y="Temp", color="Year", width=1000,
                     title="Average daily temperature change as a function of the "
                           "`DayOfYear`".title())
    fig.update_layout(title={"x": 0.5})
    fig.write_image("Q2_israel_average_daily_temperature.png")

    month_group = israel_temp.groupby("Month", as_index=False).Temp.agg("std")
    fig = px.bar(month_group, x="Month", y="Temp", width=800, height=500, text_auto='.2s',
                 title="Standard Deviation of Daily Temperatures by Month".title(),
                 labels={"Temp": "Temp Std"})
    fig.update_layout(title={"x": 0.5})
    fig.write_image("Q2_israel_std_of_Daily_Temperatures_by_Month.png")
    return israel_temp


def countries_exploring_data(temp: pd.DataFrame):
    """
    creates the plot - The average monthly temperature with error bars
    Parameters
    ----------
    temp: pd.DataFrame
            main temperature dataframe

    Returns
    -------
    None

    """
    county_month_gp = temp.groupby(["Month", "Country"], as_index=False).Temp.agg(["mean", "std"])
    county_month_gp = county_month_gp.reset_index()
    fig = px.line(county_month_gp, x="Month", y="mean", error_y="std", color="Country",
                  width=1000, height=500, title="The average monthly temperature with error "
                                                "bars".title())
    fig.update_layout(title={"x": 0.5})
    fig.write_image("Q3_countries_average_monthly_temperature.png")


def fitting_model_different_k(x: pd.DataFrame) -> pd.DataFrame:
    """
    creates the plot - Test error recorded for each value of k
    Parameters
    ----------
    x: pd.DataFrame
        israel temperature dataframe

    Returns
    -------
    results_df: pd.DataFrame
        dataframe containing the loss of each k by the given x
    """
    train_x, train_y, test_x, test_y = split_train_test(x["DayOfYear"], x["Temp"])

    results = np.zeros((10, 2))
    for k in range(1, 11):
        poly_fit = PolynomialFitting(k).fit(train_x, train_y)
        k_loss = np.round(poly_fit.loss(test_x, test_y), 2)
        results[k - 1, 1], results[k - 1, 0] = k_loss, k
        print(f"given k={k} the loss is: {k_loss}")

    results_df = pd.DataFrame(results, columns=["k", "Loss"])
    fig = px.bar(results_df, x="k", y="Loss", labels={"Loss": "Test loss"}, text_auto=True,
                 title="Test error recorded for each value of k".title())
    fig.update_layout(title={"x": 0.5})
    fig.write_image("Q4_fitting_model_for_different_k.png")
    return results_df


def evaluating_fitted_model_on_different_countries(israel_loss: pd.DataFrame,
                                                   israel_temp: pd.DataFrame, df: pd.DataFrame):
    """
    create the plot - Evaluating Loss of Israel fitted model on different countries
    Parameters
    ----------
    israel_loss: pd.DataFrame
        dataframe containing the loss of each k by the given x
    israel_temp: pd.DataFrame
        main temperature dataframe only for israel
    df: pd.DataFrame
        main temperature dataframe

    Returns
    -------
    None

    """
    # get the minimal k
    min_k_row = israel_loss["Loss"].idxmin()
    min_k = int(israel_loss.loc[min_k_row, "k"])

    model = PolynomialFitting(min_k).fit(israel_temp["DayOfYear"], israel_temp["Temp"])
    results = []
    for country in df["Country"].unique():
        if country != "Israel":
            country_df = df[df["Country"] == country]
            results.append([country, model.loss(country_df["DayOfYear"], country_df["Temp"])])

    results_df = pd.DataFrame(results, columns=['Country', 'Loss'])
    fig = px.bar(results_df, x="Country", y="Loss", text_auto=True,
                 title="Evaluating Loss of Israel fitted model on different countries".title())
    fig.update_layout(title={"x": 0.5})
    fig.write_image("Q5_evaluating_fitted_model_on_different_countries.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    temp = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_temp = israel_exploring_data(temp)

    # Question 3 - Exploring differences between countries
    countries_exploring_data(temp)

    # Question 4 - Fitting model for different values of `k`
    israel_loss = fitting_model_different_k(israel_temp)

    # Question 5 - Evaluating fitted model on different countries
    evaluating_fitted_model_on_different_countries(israel_loss, israel_temp, temp)
