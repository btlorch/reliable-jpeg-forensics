from detector.out_of_distribution_detector import OutOfDistributionDetector
from sklearn.metrics import accuracy_score, roc_auc_score
from bayesian_logistic_regression.variational_approximation import VariationalLogisticRegressionIsotropicPrior, VariationalLogisticRegressionFullCovariancePrior
from sklearn.preprocessing import StandardScaler
import numpy as np


class VariationalLogisticRegressionOutOfDistributionDetector(OutOfDistributionDetector):
    def __init__(self, clf, scale_mean=False, scale_std=False):
        super(VariationalLogisticRegressionOutOfDistributionDetector, self).__init__()

        self._scaler = StandardScaler(with_mean=scale_mean, with_std=scale_std)
        self._clf = clf
        self._X_test = None
        self._y_test = None
        self._inter_class_distance = None

    def fit(self, X_train, y_train, X_test, y_test, verbose=False):
        # Pre-processing
        self._scaler.fit(X_train)

        self._clf.fit(self._scaler.transform(X_train), y_train, verbose=verbose)

        # Compute inter-class distance
        # This assumes that the first and second halves of X_train contain the very same images
        assert all(y_train[:len(y_train) // 2] == 0) and all(y_train[len(y_train) // 2:] == 1), "Assumes that first half of training examples are the single-compressed images"
        self._inter_class_distance = np.mean(np.sqrt(np.sum(np.square(X_train[y_train == 0] - X_train[y_train == 1]), axis=-1)))

        # Retain test data for later
        self._X_test = X_test
        self._y_test = y_test

    def predict(self, X):
        return self._clf.predict_draws(self._scaler.transform(X))

    def in_distribution_accuracy(self):
        y_pred, y_var = self._clf.predict_draws(self._scaler.transform(self._X_test))
        return accuracy_score(self._y_test, (y_pred >= 0.5).astype(np.int))

    @property
    def inter_class_distance(self):
        return self._inter_class_distance

    def eval_ood(self, ood_X_test, ood_y_test, outlier_indicator=None):
        ind_y_pred, ind_y_var = self._clf.predict_draws(self._scaler.transform(self._X_test))
        ood_y_pred, ood_y_var = self._clf.predict_draws(self._scaler.transform(ood_X_test))

        # Compute accuracy on out-of-distribution test set
        ood_accuracy = accuracy_score(ood_y_test, (ood_y_pred >= 0.5).astype(np.int))

        if outlier_indicator is None:
            outlier_indicator = np.ones_like(ood_y_var)

        # Compute detectability of out-of-distribution test set
        auc = roc_auc_score(
            y_true=np.concatenate([np.zeros_like(ind_y_var), outlier_indicator], axis=0),
            y_score=np.concatenate((ind_y_var, ood_y_var), axis=0)
        )

        return {
            "ood_accuracy": ood_accuracy,
            "auc": auc,
        }

    def additional_scores(self):
        return {
            "lower_bound": self._clf.lower_bound(),
        }


class VariationalLogisticRegressionIsotropicPriorOutOfDistributionDetector(VariationalLogisticRegressionOutOfDistributionDetector):
    def __init__(self, alpha, fit_intercept=True, scale_mean=False, scale_std=False):
        clf = VariationalLogisticRegressionIsotropicPrior(alpha=alpha, fit_intercept=fit_intercept)
        super(VariationalLogisticRegressionIsotropicPriorOutOfDistributionDetector, self).__init__(clf=clf, scale_mean=scale_mean, scale_std=scale_std)

    def additional_scores(self):
        res = super().additional_scores()
        res["alpha"] = self._clf.alpha
        return res


class VariationalLogisticRegressionFullCovariancePriorOutOfDistributionDetector(VariationalLogisticRegressionOutOfDistributionDetector):
    def __init__(self, prior_scale, fit_intercept=False, scale_mean=False, scale_std=False):
        clf = VariationalLogisticRegressionFullCovariancePrior(prior_w_var_inv=None, fit_intercept=fit_intercept)
        super(VariationalLogisticRegressionFullCovariancePriorOutOfDistributionDetector, self).__init__(clf=clf, scale_mean=scale_mean, scale_std=scale_std)

        self._prior_scale = prior_scale

    def fit(self, X_train, y_train, X_test, y_test, verbose=False):
        # Add tiny epsilon to diagonal entries to ensure positive semi-definiteness, see https://stats.stackexchange.com/questions/50947/numerical-instability-of-calculating-inverse-covariance-matrix
        prior_w_var_inv = self._prior_scale * np.cov(X_train, rowvar=False) + 1e-5 * np.eye(X_train.shape[1])
        self._clf.prior_w_var_inv = prior_w_var_inv
        super().fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, verbose=verbose)

    def additional_scores(self):
        res = super().additional_scores()
        res["prior_scale"] = self._prior_scale
        return res
