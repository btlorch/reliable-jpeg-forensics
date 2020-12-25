from abc import ABC, abstractmethod


class OutOfDistributionDetector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X_train, y_train, X_test, y_test, **fit_kwargs):
        pass

    @abstractmethod
    def in_distribution_accuracy(self):
        pass

    @abstractmethod
    def eval_ood(self, ood_X_test, ood_y_test, outlier_indicator=None):
        pass

    @abstractmethod
    def inter_class_distance(self):
        pass

    def additional_scores(self):
        return {}
