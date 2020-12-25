import numpy as np


class LogisticRegression(object):
    """
    Base class for logistic regression classifier
    """
    def __init__(self):
        pass

    def fit(self, X, t, max_iter, verbose):
        pass

    def predict(self, X):
        pass

    @staticmethod
    def sigmoid(a):
        return np.tanh(a * 0.5) * 0.5 + 0.5

    @staticmethod
    def log_sigmoid(x):
        return -np.log(1 + np.exp(-x))

    @staticmethod
    def _add_intercept(X):
        """
        Prepends a constant one feature dimension to the data
        :param X: ndarray of shape [num_samples, num_features]
        :return: ndarray of shape [num_samples, num_features + 1] with a constant one in the first column
        """
        return np.hstack((np.ones([X.shape[0], 1]), X))
