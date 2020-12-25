from sklearn.neighbors import KNeighborsClassifier
from detector.out_of_distribution_detector import OutOfDistributionDetector
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
import numpy as np


class KNearestNeighborOutOfDistributionDetector(OutOfDistributionDetector):
    def __init__(self, n_neighbors, scale_mean=False, scale_std=False):
        super(KNearestNeighborOutOfDistributionDetector, self).__init__()

        num_threads = mp.cpu_count()
        self._clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=num_threads)
        self._scaler = StandardScaler(with_mean=scale_mean, with_std=scale_std)
        self._X_test = None
        self._y_test = None
        self._inter_class_distance = None

    def fit(self, X_train, y_train, X_test, y_test, verbose=False):
        # Pre-processing
        self._scaler.fit(X_train)

        self._clf.fit(self._scaler.transform(X_train), y_train)

        # Retain test data for later
        self._X_test = X_test
        self._y_test = y_test

        # Compute inter-class distance
        # This assumes that the first and second halves of X_train contain the very same images
        assert all(y_train[:len(y_train) // 2] == 0) and all(y_train[len(y_train) // 2:] == 1), "Assumes that first half of training examples are the single-compressed images"
        self._inter_class_distance = np.mean(np.sqrt(np.sum(np.square(X_train[y_train == 0] - X_train[y_train == 1]), axis=-1)))

    def in_distribution_accuracy(self):
        return self._clf.score(self._scaler.transform(self._X_test), self._y_test)

    def eval_ood(self, ood_X_test, ood_y_test, outlier_indicator=None):
        ind_neighbor_distances, _ = self._clf.kneighbors(self._scaler.transform(self._X_test))
        # Average over n_neighbors
        ind_neighbor_distances = np.mean(ind_neighbor_distances, axis=1)

        ood_neighbor_distances, _ = self._clf.kneighbors(self._scaler.transform(ood_X_test))
        # Average over n_neighbors
        ood_neighbor_distances = np.mean(ood_neighbor_distances, axis=1)

        # Compute accuracy on out-of-distribution test set
        ood_accuracy = self._clf.score(self._scaler.transform(ood_X_test), ood_y_test)

        if outlier_indicator is None:
            outlier_indicator = np.ones_like(ood_neighbor_distances)

        auc = roc_auc_score(
            y_true=np.concatenate([np.zeros_like(ind_neighbor_distances), outlier_indicator], axis=0),
            y_score=np.concatenate((ind_neighbor_distances, ood_neighbor_distances), axis=0)
        )

        return {
            "ood_accuracy": ood_accuracy,
            "auc": auc,
        }

    def additional_scores(self):
        return {
            "n_neighbors": self._clf.n_neighbors
        }

    @property
    def inter_class_distance(self):
        return self._inter_class_distance
