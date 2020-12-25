from decoder import PyCoefficientDecoder
import numpy as np


FREQ_BANDS = (1, 8, 16, 9, 2, 3, 10, 17, 24) # AC bands in zig-zag order
FULL_RESOLUTION = "full_resolution"


def extract_features_from_signal(signal):
    """
    :param signal: unquantized DCT coefficients, integer values
    :return: histogram, counting the number of occurrences of each of the nine digits (0 cannot be a significant digit)
    """

    # Obtain first significant digit, exclude the value 0
    mask = signal != 0
    number_of_digits = np.floor(np.log10(np.abs(signal[mask]))).astype(np.int) + 1
    first_digit = np.abs(signal[mask]) // np.power(10, number_of_digits - 1)

    # Alternative way to obtain first digits via string conversion
    alternative = np.array([int(str(np.abs(x))[0]) for x in signal if x != 0])
    assert np.allclose(alternative, first_digit)

    # Create histogram over first digit
    hist, bin_edges = np.histogram(first_digit, bins=9, range=(1, 10))
    return hist


def extract_features_from_file_full_res(filename, freq_bands=FREQ_BANDS):
    """
    Extract features from given file
    :param filename: path to JPEG image
    :param freq_bands: by default, take the first 9 spatial frequencies in zig-zag order
    :return: features of shape [num_freq_bands, 9]
    """

    d = PyCoefficientDecoder(filename)

    # Retrieve DCT coefficients
    dct_coefs = d.get_dct_coefficients(0)

    # Load quantization table
    quant_table = d.get_quantization_table(0).flatten()

    # Un-quantize DCT coefficients
    dct_coefs = dct_coefs * quant_table

    # Extract first-digit features for each frequency band
    features = []
    for freq_band in freq_bands:
        features.append(extract_features_from_signal(dct_coefs[:, freq_band]))

    features = np.stack(features, axis=0)
    return features


def extract_features_from_file(filename, freq_bands=FREQ_BANDS, crops=None):
    """
    Extract features from a given file, crop to several sizes
    :param filename: path to JPEG image
    :param freq_bands: by default, take the first 9 spatial frequencies in zig-zag order
    :param crops: list of crop sizes in pixels. If crops is `None` or if `None` is one of the list items, extract features from the full-resolution image.
    :return: features of shape [num_crops, num_freq_bands, 9], or [num_freq_bands, 9] if crops is `None`
    """

    if crops is None:
        return extract_features_from_file_full_res(filename=filename, freq_bands=freq_bands)

    d = PyCoefficientDecoder(filename)

    # Retrieve DCT coefficients
    dct_coefs = d.get_dct_coefficients(0)

    # Load quantization table
    quant_table = d.get_quantization_table(0).flatten()

    # Un-quantize DCT coefficients
    dct_coefs = dct_coefs * quant_table

    # Reshape blocks into image structure
    height_in_blocks = d.get_height_in_blocks(0)
    width_in_blocks = d.get_width_in_blocks(0)

    dct_coefs_aligned = dct_coefs.reshape(height_in_blocks, width_in_blocks, 64)

    features = []

    # Iterate over crop sizes
    for crop in crops:
        # Keep a list of features for different frequency bands of the current crop
        crop_features = []

        # Special string for full resolution
        if crop == FULL_RESOLUTION:
            dct_coefs_cropped = dct_coefs

        else:
            assert crop <= height_in_blocks * 8 and crop <= width_in_blocks * 8, "Requested crop size is larger than image size"
            assert crop % 8 == 0, "Crop sizes must be multiples of 8"

            # Convert requested size into number of blocks
            crop_height_in_blocks = crop // 8
            crop_width_in_blocks = crop // 8

            # Calculate offset of crop from upper left
            crop_block_offset_y = (height_in_blocks - crop_height_in_blocks) // 2
            crop_block_offset_x = (width_in_blocks - crop_width_in_blocks) // 2

            # Crop
            dct_coefs_aligned_cropped = dct_coefs_aligned[crop_block_offset_y:crop_block_offset_y + crop_height_in_blocks, crop_block_offset_x:crop_block_offset_x + crop_width_in_blocks]

            # To display the crop at this point, convert the DCT coefficients into image space
            # from scipy.fftpack import idct
            # blocks_8x8 = np.apply_along_axis(lambda x: idct(idct(x.reshape(8, 8), axis=1, norm="ortho"), axis=0, norm="ortho"), axis=2, arr=dct_coefs_aligned_cropped)
            # img = np.transpose(blocks_8x8, axes=[0, 2, 1, 3]).reshape(crop_height_in_blocks * 8, crop_width_in_blocks * 8)

            # Flatten
            dct_coefs_cropped = dct_coefs_aligned_cropped.reshape(crop_height_in_blocks * crop_width_in_blocks, 64)

        # Extract features for each frequency band
        for freq_band in freq_bands:
            crop_features.append(extract_features_from_signal(dct_coefs_cropped[:, freq_band]))

        features.append(crop_features)

    return np.stack(features, axis=0)
