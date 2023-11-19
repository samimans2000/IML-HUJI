from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    mu = 10
    sigma = 1
    m = 1000

    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(loc=mu, scale=sigma, size=m)
    x_fit = UnivariateGaussian(biased_var=False).fit(samples)
    print(f"({round(x_fit.mu_, 3)}, {round(x_fit.var_, 3)})")

    # Question 2 - Empirically showing sample mean is consistent
    increasing_factor = 10
    distance_estimate_and_true = []
    for i in range(1, int(m / increasing_factor) + 1):
        fit = UnivariateGaussian(biased_var=False).fit(samples[:(i * increasing_factor)])
        distance_estimate_and_true.append(abs(mu - fit.mu_))

    fig_q2 = go.Figure(
        go.Scatter(x=[i * increasing_factor for i in range(1, int(m / increasing_factor) + 1)],
                   y=distance_estimate_and_true,
                   mode='markers', marker=dict(color="black")), layout=go.Layout(
            title=dict(text=r"$\text{Estimation of Distance between True and Estimated "
                            r"Means with Increasing Sample Size}$", x=0.5),
            xaxis_title=dict(text=r"$\text{m - number of samples}$"),
            yaxis_title=dict(text=r"$\text{sample mean estimator distance }\hat{\mu}_m$$"),
            height=400, width=1000))
    fig_q2.write_image("Q2_empirically_showing_sample_mean_is_consistent.png")

    # Question 3 - Plotting Empirical PDF of fitted model
    x_pdf = x_fit.pdf(samples)
    fig_q3 = go.Figure(
        go.Scatter(x=samples,
                   y=x_pdf,
                   mode='markers', marker=dict(color="black")), layout=go.Layout(
            title=dict(text=r"$\text{Plotting Empirical PDF of fitted model}$", x=0.5),
            xaxis_title=dict(text=r"$\text{x}$"),
            yaxis_title=dict(text=r"$\text{PDF(x)}$"),
            height=400, width=1000))
    fig_q3.write_image("Q3_plotting_empirical_pdf_of_fitted_model.png")


def test_multivariate_gaussian():
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    m = 1000

    # Question 4 - Draw samples and print fitted model
    samples = np.random.multivariate_normal(mean=mu, cov=cov, size=m)
    x_fit = MultivariateGaussian().fit(samples)
    print(np.round(x_fit.mu_, 3))
    print(np.round(x_fit.cov_, 3))

    # Question 5 - Likelihood evaluation
    f_values = np.linspace(-10, 10, 200)
    mu_log_likelihood = []
    for i in range(f_values.size):
        mu_f_1 = []
        for j in range(f_values.size):
            mu_ = np.array([f_values[i], 0, f_values[j], 0])
            mu_f_1.append(MultivariateGaussian.log_likelihood(mu_, cov, samples))
        mu_log_likelihood.append(mu_f_1)
    mu_matrix = np.array(mu_log_likelihood)

    fig_q5 = go.Figure(data=go.Heatmap(x=f_values, y=f_values, z=mu_matrix,
                                       colorscale='agsunset_r'))
    fig_q5.update_layout(
        title=dict(text=r"$\text{Log-Likelihood values for Multivariate Gaussian with "
                        r"expectation of }\mu[0],\mu[2]$", x=0.5),
        xaxis_title=dict(text=r"$\mu[2] = f_3$"),
        yaxis_title=dict(text=r"$\mu[0] = f_1$"),
        height=400, width=900)
    fig_q5.write_image("Q5_likelihood_evaluation.png")

    # Question 6 - Maximum likelihood
    max_val_num = np.argmax(mu_matrix)
    max_indexes = np.unravel_index(max_val_num, mu_matrix.shape)  # get specific index
    f1_max, f3_max = round(f_values[max_indexes[0]], 3), round(f_values[max_indexes[1]], 3)
    print(f"Model achieved the maximum log-likelihood value -> (f1, f3)=({f1_max}, {f3_max})")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
