from bayesian_logistic_regression.logistic_regression import LogisticRegression
from scipy.special import loggamma
from utils.linalg import symmetric_inv
import numpy as np


class VariationalLogisticRegressionIsotropicPrior(LogisticRegression):
    """
    Implementation adapted from https://github.com/ctgk/PRML/blob/master/prml/linear/variational_logistic_regression.py
    Equation numbers refer to Bishop's "Pattern Recognition and Machine Learning", Chapter 10.
    """
    def __init__(self, alpha=None, a0=1., b0=1., fit_intercept=True):
        """
        Set up variational logistic regression model with isotropic prior, i.e.:
            p(w | alpha) = N(w | 0, alpha^{-1} I)

        If alpha is not set, alpha is learned from the training data. In this case, we assume as gamma hyper-prior over alpha, i.e.:
            p(alpha) = Gamma(alpha | a0, b0)

        The posterior over alpha results in another gamma distribution:
            p(alpha) = Gamma(alpha | a, b)

        :param alpha: precision parameter of the prior. If None, alpha is estimated from the data.
        :param a0: parameter of the gamma prior distribution over alpha
        :param b0: parameter of the gamma prior distribution over alpha
        :param fit_intercept: flag whether to add an intercept term
        """
        super(VariationalLogisticRegressionIsotropicPrior, self).__init__()

        if alpha is not None:
            self._alpha = alpha
        else:
            self.a0 = a0
            self.b0 = b0

        self._fit_intercept = fit_intercept

        # Parameters set by fit()
        self.w_var_inv = None # Variational posterior precision
        self.w_var = None # Variational posterior covariance
        self.w_mean = None # Variational posterior mean
        self.xi = None # Variational parameters

    def fit(self, X, t, max_iter=1000, verbose=True):
        """
        Fit variational posterior distribution over weights
        :param X: training data of shape [num_samples, num_features]
        :param t: training labels of shape [num_samples], must be in {0, 1}
        :param max_iter: maximum number of training iterations
        :param verbose: flag whether to print lower bound after each EM step
        """

        # Make labels consistent with X.dtype
        t = np.asarray(t, dtype=X.dtype)

        # Prepend intercept dimension if desired
        if self._fit_intercept:
            X = self._add_intercept(X)

        N, M = X.shape

        # If alpha is not fixed, calculate posterior over alpha
        if hasattr(self, "a0"):
            self.a = self.a0 + 0.5 * M

        # Initialize variational parameters
        self.xi = np.ones(N)

        # Keep a copy of the current set of variational parameters
        param = np.copy(self.xi)

        for i in range(max_iter):
            # E-step: Given the current variational parameters, determine variational posterior
            # Eq. (10.176): Covariance of variational posterior. Note that the original implementation differs in the prior covariance and lambda.
            self.w_var_inv = 2 * (self._lambda(self.xi) * X.T) @ X
            alpha_vec = np.ones(M) * self.alpha
            if self._fit_intercept:
                # Borrowed from *sklearn_bayes*: No regularization for intercept term
                alpha_vec[0] = np.finfo(np.float32).eps

            np.fill_diagonal(self.w_var_inv, np.diag(self.w_var_inv) + alpha_vec)
            self.w_var = symmetric_inv(self.w_var_inv)

            # Eq. (10.175): Mean of variational posterior
            self.w_mean = self.w_var @ np.sum(X.T * (t - 0.5), axis=1)

            # M-step: Maximize complete-data log likelihood, that is, update variational parameters to lower bound
            # Eq. (10.163)
            self.xi = np.sqrt(np.sum(X @ (self.w_var + self.w_mean * self.w_mean[:, None]) * X, axis=-1))

            # Compute value of lower bound
            if verbose:
                lower_bound = self.lower_bound()
                print(("[{:0" + str(int(np.ceil(np.log10(max_iter))) + 1) + "d}] Lower bound = {:4.3f}").format(i, lower_bound))

            # Stop if the variational parameters did not change significantly
            if np.allclose(self.xi, param):
                break
            else:
                param = np.copy(self.xi)

    @staticmethod
    def _lambda(xi):
        # Eq. (10.150)
        # This is equivalent to 1. / (2. * xi) * (self.sigmoid(xi) - 0.5)
        # Prevent division by zero
        return np.divide(np.tanh(xi * 0.5) * 0.25, xi, out=np.zeros_like(xi), where=~np.isclose(xi, 0))

    def lower_bound(self):
        """
        Compute the lower bound on the log marginal likelihood
        Analytic integration over w with Gaussian prior p(w) and h(w, xi) that is the exponential of a quadratic function of w.
        xi are the variational parameters, one for each training sample.
        :return: value of lower bound
        """
        M = len(self.w_mean)

        # If alpha is fixed
        if hasattr(self, "_alpha"):
            # Closed form solution given by Eq. (10.164)
            l1 = 0.5 * (np.linalg.slogdet(self.w_var)[1] + M * np.log(self.alpha))
            l2 = 0.5 * (self.w_mean @ self.w_var_inv @ self.w_mean)
            l3 = np.sum(self.log_sigmoid(self.xi) - 0.5 * self.xi + self._lambda(self.xi) * (self.xi ** 2))

            return l1 + l2 + l3

        # Otherwise the lower bound also contains the hyper-prior
        # See https://arxiv.org/pdf/1310.5438.pdf, Equations (56) - (62)
        l1 = 0.5 * (self.w_mean @ self.w_var_inv @ self.w_mean)
        l2 = 0.5 * np.linalg.slogdet(self.w_var)[1]
        l3 = np.sum(self.log_sigmoid(self.xi) - 0.5 * self.xi + self._lambda(self.xi) * (self.xi ** 2))
        l4 = -loggamma(self.a0) + self.a0 * np.log(self.b0) - self.b0 * self.a / self.b - self.a * np.log(self.b) + loggamma(self.a) + self.a

        return l1 + l2 + l3 + l4

    @property
    def alpha(self):
        if hasattr(self, "_alpha"):
            # If alpha is given, use alpha
            return self._alpha

        elif self.w_mean is None:
            # alpha is not given, and we have not calculated the weight posterior yet
            # Hence, take b of prior.
            self.b = self.b0

        else:
            # Calculate posterior
            # Eq. (10.179)
            if self._fit_intercept:
                self.b = self.b0 + 0.5 * (np.sum(self.w_mean[1:] ** 2) + np.trace(self.w_var[1:, 1:]))
            else:
                self.b = self.b0 + 0.5 * (np.sum(self.w_mean ** 2) + np.trace(self.w_var))

        return self.a / self.b

    def predict(self, X):
        """
        Compute posterior probability that an input belongs to class 1, i.e., p(C_1 | x).
        :param X: test data of shape [num_samples, num_features]
        :return: posterior probabilities p(C_1 | x) of shape [num_samples]
        """
        if self._fit_intercept:
            X = self._add_intercept(X)

        mu_a = X @ self.w_mean
        var_a = np.sum(X @ self.w_var * X, axis=1)
        y = self.sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))
        return y

    def predict_draws(self, X, num_monte_carlo_draws=100):
        """
        :param X: ndarray of shape [num_samples, num_features]
        :param num_monte_carlo_draws: number of draws from the weight posterior
        :return: (y, y_var)
            y is the mean over the Monte Carlo draws
            y_var is the variance the Monte Carlo draws
        """
        if self._fit_intercept:
            X = self._add_intercept(X)

        w_draws = np.random.multivariate_normal(mean=self.w_mean, cov=self.w_var, size=num_monte_carlo_draws)
        a_draws = X @ w_draws.T
        y_draws = self.sigmoid(a_draws)

        y = np.mean(y_draws, axis=1)
        y_var = np.var(y_draws, axis=1)

        return y, y_var


class VariationalLogisticRegressionFullCovariancePrior(LogisticRegression):
    def __init__(self, prior_w_var_inv, fit_intercept=True):
        """
        Set up variational logistic regression model with full covariance prior
        :param prior_w_var_inv: prior precision matrix (inverse of prior covariance). The precision matrix must be of shape [m, m] where m is the number of feature dimensions (+ 1 if fit_intercept=True).
        :param fit_intercept: flag whether to add an intercept term
        """
        super(VariationalLogisticRegressionFullCovariancePrior, self).__init__()

        self.prior_w_var_inv = prior_w_var_inv
        self.prior_w_var = None # Only needed to compute lower bound
        self._fit_intercept = fit_intercept

        # Parameters set by fit()
        self.w_var_inv = None # Variational posterior precision
        self.w_var = None # Variational posterior covariance
        self.w_mean = None # Variational posterior mean
        self.xi = None # Variational parameters

    def fit(self, X, t, max_iter=1000, verbose=True):
        """
        Fit variational posterior distribution
        :param X: training data of shape [num_samples, num_features]
        :param t: training labels of shape [num_samples], must be in {0, 1}
        :param max_iter: maximum number of training iterations
        :param verbose: flag whether to print lower bound after each EM step
        """

        # Make labels consistent with X.dtype
        t = np.asarray(t, dtype=X.dtype)

        # Prepend intercept dimension if desired
        if self._fit_intercept:
            X = self._add_intercept(X)

        N, M = X.shape
        assert M == self.prior_w_var_inv.shape[0], "Dimension mismatch between features and precision matrix"

        # Initialize variational parameters
        self.xi = np.ones(N)

        # Obtain prior covariance from precision matrix
        self.prior_w_var = symmetric_inv(self.prior_w_var_inv)

        # Keep a copy of the current set of variational parameters
        param = np.copy(self.xi)

        for i in range(max_iter):
            # E-step: Given the current variational parameters, determine variational posterior
            # Eq. (10.176): Covariance of variational posterior. Note that the original implementation differs in the prior covariance and lambda.
            self.w_var_inv = self.prior_w_var_inv + 2 * (self._lambda(self.xi) * X.T) @ X
            self.w_var = symmetric_inv(self.w_var_inv)

            # Eq. (10.175): Mean of variational posterior
            self.w_mean = self.w_var @ np.sum(X.T * (t - 0.5), axis=1)

            # M-step: Maximize complete-data log likelihood, that is, update variational parameters to lower bound
            # Eq. (10.163)
            self.xi = np.sqrt(np.sum(X @ (self.w_var + self.w_mean * self.w_mean[:, None]) * X, axis=-1))

            # Compute value of lower bound
            if verbose:
                lower_bound = self.lower_bound()
                print(("[{:0" + str(int(np.ceil(np.log10(max_iter))) + 1) + "d}] Lower bound = {:4.3f}").format(i, lower_bound))

            # Stop if the variational parameters did not change significantly
            if np.allclose(self.xi, param):
                break
            else:
                param = np.copy(self.xi)

    @staticmethod
    def _lambda(xi):
        # Eq. (10.150)
        # This is equivalent to 1. / (2. * xi) * (self.sigmoid(xi) - 0.5)
        # Prevent division by zero
        return np.divide(np.tanh(xi * 0.5) * 0.25, xi, out=np.zeros_like(xi), where=~np.isclose(xi, 0))

    def lower_bound(self):
        """
        Compute the lower bound on the log marginal likelihood
        Analytic integration over w with Gaussian prior p(w) and h(w, xi) that is the exponential of a quadratic function of w.
        xi are the variational parameters, one for each training example.
        :return: value of lower bound
        """

        # Closed form solution given by Eq. (10.164)
        l1 = 0.5 * (np.linalg.slogdet(self.w_var)[1] - np.linalg.slogdet(self.prior_w_var)[1])
        l2 = 0.5 * (self.w_mean @ self.w_var_inv @ self.w_mean)
        l3 = np.sum(self.log_sigmoid(self.xi) - 0.5 * self.xi + self._lambda(self.xi) * (self.xi ** 2))

        return l1 + l2 + l3

    def predict(self, X):
        """
        Compute posterior probability that an input belongs to class 1, i.e., p(C_1 | x).
        :param X: test data of shape [num_samples, num_features]
        :return: posterior probabilities p(C_1 | x) of shape [num_samples]
        """
        if self._fit_intercept:
            X = self._add_intercept(X)

        mu_a = X @ self.w_mean
        var_a = np.sum(X @ self.w_var * X, axis=1)
        y = self.sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))
        return y

    def predict_draws(self, X, num_monte_carlo_draws=100):
        """
        :param X: ndarray of shape [num_samples, num_features]
        :param num_monte_carlo_draws: number of draws from the weight posterior
        :return: (y, y_var)
            y is the mean over the Monte Carlo draws
            y_var is the variance over the Monte Carlo draws
        """
        if self._fit_intercept:
            X = self._add_intercept(X)

        # Add tiny constant to avoid a numerical issue "On entry to DLASCLS parameter number 4 had an illegal value"
        cov = self.w_var + 1e-6 * np.eye(len(self.w_var))
        w_draws = np.random.multivariate_normal(mean=self.w_mean, cov=cov, size=num_monte_carlo_draws)
        a_draws = X @ w_draws.T
        y_draws = self.sigmoid(a_draws)

        y = np.mean(y_draws, axis=1)
        y_var = np.var(y_draws, axis=1)

        return y, y_var
