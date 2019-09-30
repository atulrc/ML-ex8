# Scientific and vector computation for python
import numpy as np

def estGaus(X):
    """
    This function estimates the parameters of a Gaussian distribution
    using a provided dataset.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n) with each n-dimensional
        data point in one row, and each total of m data points.

    Returns
    -------
    mu : array_like
        A vector of shape (n,) containing the means of each dimension.

    sigma2 : array_like
        A vector of shape (n,) containing the computed
        variances of each dimension.

    Instructions
    ------------
    Compute the mean of the data and the variances
    In particular, mu[i] should contain the mean of
    the data for the i-th feature and sigma2[i]
    should contain variance of the i-th feature.
    """
    # Useful variables
    m, n = X.shape

    # You should return these values correctly
    mu = np.zeros(n)
    sigma2 = np.zeros(n)

    # ====================== YOUR CODE HERE ======================

    mu = (1 / m) * np.sum(X, axis = 0)
    sigma2 = (1 / m) * np.sum((X - mu) ** 2, axis = 0)

    # =============================================================
    return mu, sigma2
