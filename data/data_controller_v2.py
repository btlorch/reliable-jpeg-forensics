from data.benfords_law_features import FULL_RESOLUTION
from data.load_dataset import normalize_histograms
import pandas as pd
import numpy as np
import collections
import itertools
import h5py
import os


QF_PAIR_PATTERN = "qf1_{:d}_qf2_{:d}"


class DataControllerV2(object):
    def __init__(self, data_file, use_cache=False):
        self._data_file = data_file
        self._f = None

        self._cache = None
        if use_cache:
            self._cache = self._fill_cache()

    def _fill_cache(self):
        cache = {}
        with h5py.File(self._data_file, "r") as f:

            def load_datasets(name, item):
                is_dataset = isinstance(item, h5py.Dataset)
                if is_dataset:
                    cache[name] = item[()]

            f.visititems(load_datasets)

        return cache

    def __enter__(self):
        self._f = h5py.File(self._data_file, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._f is not None:
            self._f.close()
        self._f = None

    def get_filenames(self, qf1=95, qf2=95):
        path = os.path.join(QF_PAIR_PATTERN.format(qf1, qf2), FULL_RESOLUTION, "filename")
        # Read from cache
        if self._cache is not None:
            return np.sort(self._cache[path])

        # Read from file
        return np.sort(self._f[path][()])

    def _load_dataset(self, qf1, qf2, crop):
        path = os.path.join(QF_PAIR_PATTERN.format(qf1, qf2), str(crop))

        if self._cache is None:
            # Read from file
            group = self._f[path]
            filenames = group["filename"][()]
            X_sc = group["sc_features"][()]
            X_dc = group["dc_features"][()]
        else:
            # Read from cache
            filenames = self._cache[os.path.join(path, "filename")]
            X_sc = self._cache[os.path.join(path, "sc_features")]
            X_dc = self._cache[os.path.join(path, "dc_features")]

        # Ensure that files are in sorted order
        assert all([a == b for a, b in zip(list(sorted(filenames)), filenames)])

        # Normalize such that the sum over each histogram is 1
        # Avoid division by zero (for histograms with only empty bins from flat image patches)
        X_sc = normalize_histograms(X_sc)
        X_dc = normalize_histograms(X_dc)

        # Flatten last two dimensions
        X_sc = X_sc.reshape(len(X_sc), -1)
        X_dc = X_dc.reshape(len(X_dc), -1)

        return filenames, X_sc, X_dc

    def _get_data(self, qf1s, qf2s, subsets=(None,), crop=FULL_RESOLUTION, drop_duplicates=True, stratify=True, random_state=None):
        assert isinstance(subsets, collections.Sequence), "Given `subsets` must be a sequence"

        # Convert qf1s and qf2s to lists
        if not isinstance(qf1s, collections.Sequence):
            qf1s = [qf1s]

        if not isinstance(qf2s, collections.Sequence):
            qf2s = [qf2s]

        X_buffer = []
        meta_buffer = []

        # Keep an index column that references into X_buffer. Otherwise we would concatenate buffers with the same indices.
        next_index = 0

        for qf1, qf2 in itertools.product(qf1s, qf2s):
            filenames, X_sc, X_dc = self._load_dataset(qf1=qf1, qf2=qf2, crop=crop)

            num_sc_samples = len(X_sc)
            num_dc_samples = len(X_dc)
            assert num_sc_samples == num_dc_samples

            # Single-compressed -> label 0
            sc_labels = np.zeros(num_sc_samples, dtype=np.int)
            # Double-compressed -> label 1
            dc_labels = np.ones(num_dc_samples, dtype=np.int)
            labels = np.concatenate((sc_labels, dc_labels))

            meta_buffer.append(pd.DataFrame({
                "label": labels,
                "filename": np.concatenate([filenames, filenames], axis=0),
                "qf1": [None] * num_sc_samples + [qf1] * num_dc_samples,
                "qf2": [qf2] * (num_sc_samples + num_dc_samples),
                "index": np.arange(next_index, next_index + num_sc_samples + num_dc_samples)
            }))
            X_buffer.append(X_sc)
            X_buffer.append(X_dc)
            next_index += num_sc_samples + num_dc_samples

        X = np.concatenate(X_buffer, axis=0)
        meta_df = pd.concat(meta_buffer)
        meta_df.set_index("index", inplace=True)

        sc_meta_df = meta_df[meta_df["label"] == 0]
        dc_meta_df = meta_df[meta_df["label"] == 1]

        if drop_duplicates:
            # The same single-compressed images are included several times. Make sure to drop duplicates.
            sc_meta_df = sc_meta_df.drop_duplicates()
            dc_meta_df = dc_meta_df.drop_duplicates()

        ret = ()

        for subset in subsets:
            subset_sc_meta_df = sc_meta_df.copy()
            subset_dc_meta_df = dc_meta_df.copy()

            # If subset is None, take all images
            if subset is not None:
                subset_sc_meta_df = subset_sc_meta_df[subset_sc_meta_df["filename"].isin(subset)]
                subset_dc_meta_df = subset_dc_meta_df[subset_dc_meta_df["filename"].isin(subset)]

            if stratify:
                min_num_samples = min(len(subset_sc_meta_df), len(subset_dc_meta_df))

                if len(subset_sc_meta_df) > min_num_samples:
                    subset_sc_meta_df = subset_sc_meta_df.sample(min_num_samples, replace=False, random_state=random_state)
                if len(subset_dc_meta_df) > min_num_samples:
                    subset_dc_meta_df = subset_dc_meta_df.sample(min_num_samples, replace=False, random_state=random_state)

            sc_indices = subset_sc_meta_df.index
            dc_indices = subset_dc_meta_df.index
            both_indices = np.concatenate((sc_indices, dc_indices), axis=0)
            X_subset = X[both_indices]
            y = np.concatenate((subset_sc_meta_df["label"].values, subset_dc_meta_df["label"].values), axis=0)

            ret = ret + (X_subset, y)

        return ret

    def get_all_data(self, qf1s, qf2s, crop=FULL_RESOLUTION, drop_duplicates=True, stratify=True, random_state=None):
        return self._get_data(qf1s=qf1s,
                              qf2s=qf2s,
                              subsets=(None,),
                              crop=crop,
                              drop_duplicates=drop_duplicates,
                              stratify=stratify,
                              random_state=random_state)

    def get_data(self, qf1s, qf2s, crop=FULL_RESOLUTION, train_filenames=None, test_filenames=None, drop_duplicates=True, stratify=True, random_state=None):
        """
        Note that the results are not sorted by filename
        :param qf1s: list of quality factors used for first compression
        :param qf2s: list of quality factors used for second compression
        :param crop: list of crop sizes to retrieve
        :param train_filenames: list of filenames of the training images
        :param test_filenames: list of filenames of the test images
        :param drop_duplicates: flag whether to drop duplicates
        :param stratify: if True, ensure that single- and double-compressed images are represented equally
        :param random_state: random number generator state for reproducibility
        :return: list of up to two tuples (X_subset, y_subset), one containing the features and labels corresponding to the train filenames and one for test filenames, if given.
        """
        assert train_filenames is not None or test_filenames is not None, "Either `train_filenames` or `test_filenames` must be given"

        subsets = []
        if train_filenames is not None:
            subsets.append(train_filenames)
        if test_filenames is not None:
            subsets.append(test_filenames)

        return self._get_data(qf1s=qf1s,
                              qf2s=qf2s,
                              subsets=subsets,
                              crop=crop,
                              drop_duplicates=drop_duplicates,
                              stratify=stratify,
                              random_state=random_state)
