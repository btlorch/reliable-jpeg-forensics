from sklearn.model_selection import ShuffleSplit


class FilenameShuffleSplit(ShuffleSplit):
    """
    Custom shuffle split that yields filenames instead of their indices
    """
    def _iter_indices(self, X, y=None, groups=None):
        """
        :param X: ndarray
        """
        for ind_train, ind_test in super(FilenameShuffleSplit, self)._iter_indices(X=X, y=y, groups=groups):
            yield X[ind_train], X[ind_test]
