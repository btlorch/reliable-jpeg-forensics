import pandas as pd
import numpy as np
import h5py
import os


def load_filenames(dataset_filename):
    """
    Load filenames from HDF5 file
    :param dataset_filename: path to HDF5 file containing dataset named "filename"
    :return: list of filenames
    """
    with h5py.File(dataset_filename, "r") as f:
        return list(f["filename"])


def normalize_histograms(X):
    """
    Normalize each histogram, given in the last dimension of the input
    :param X: ndarray of shape [num_freq_bands, 9] or [batch_size, num_freq_bands, 9]
    :return: normalized ndarray such that the last dimensions sums to 1 (or 0 if there were no other values than zeros)
    """
    assert X.shape[-1] == 9, "Expected 9 bins in the last dimension, one for each digit"

    # Avoid division by zero, therefore use np.divide(x1, x2, where=mask).
    # Make sure that the divisor x2 and mask have the same shape as the dividend x1.
    ndims = len(X.shape)
    # After summing over the last dimension, repeat the sum over this dimension.
    reps = [1] * (ndims - 1) + [9]
    histogram_sums = np.tile(np.sum(X, axis=-1, keepdims=True), reps)
    assert histogram_sums.shape == X.shape, "np.divide uses strange broadcasting rules"

    # Prevent division by zero
    X_normalized = np.divide(X, histogram_sums, out=np.zeros(X.shape, dtype=np.float), where=histogram_sums > 0)
    assert np.all(np.isclose(np.sum(X_normalized, axis=-1), 1.0) | np.isclose(np.sum(X_normalized, axis=-1), 0)), "All normalized histograms must sum to 1 or 0"

    return X_normalized


def load_dataset_no_split(filename):
    """
    Load features from HDF5 file
    :param filename: path to HDF5 file
    :return: (filenames, X_sc, X_dc)
        filenames: list of filenames
        X_sc: features of single-compressed images
        X_dc: features of double-compressed images
    """
    with h5py.File(filename, "r") as f:
        filenames = list(f["filename"])
        X_sc = f["sc_features"][()]
        X_dc = f["dc_features"][()]

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


def load_data_multiple_qf1(filename_pattern, qf1s, qf2, data_dir=None):
    """
    Concatenate datasets from multiple qf1 and a fixed qf2
    :param filename_pattern: filename with placeholders for qf1 and qf2
    :param qf1s: list of quality factors
    :param qf2: scalar quality factor
    :param data_dir: directory where HDF5 files are located
    :return: (X, meta_df)
        X: ndarray of features for both single- and double-compressed images
        meta_df: data frame that contains one entry per row in X. The corresponding index corresponds to the order in X. The following columns are available:
            - label: 0 for single-compressed, 1 for double-compressed images
            - filename: filename of original image
            - qf1: quality factor of first compression, or None if image was compressed only once
    """
    X_buffer = []
    meta_buffer = []

    # Keep an index column that references into X_buffer. Otherwise we would concatenate buffers with the same indices.
    next_index = 0

    for qf1 in qf1s:
        # Load single- and double-compressed images with given qf1 and qf2
        filepath = filename_pattern.format(qf1, qf2)
        if data_dir is not None:
            filepath = os.path.join(data_dir, filepath)

        filenames, X_sc, X_dc = load_dataset_no_split(filepath)

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
            "filename": filenames + filenames,
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

    return X, meta_df
