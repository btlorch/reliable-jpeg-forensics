import itertools
import numpy as np
from PIL import Image
import imageio
import tempfile
import h5py
from tqdm.auto import tqdm
import functools
import multiprocessing as mp
import argparse
import time
from glob import glob
import os
from utils.logger import setup_custom_logger
from utils.encoder import img_cjpeg, djpeg_cjpeg
from data.benfords_law_features import extract_features_from_file, FREQ_BANDS, FULL_RESOLUTION
from decoder import PyCoefficientDecoder


log = setup_custom_logger(os.path.basename(__file__))


def verify_qtable(filename, qtable):
    d = PyCoefficientDecoder(filename)
    return np.allclose(d.get_quantization_table(0), qtable)


def open_tif(filepath):
    # Pillow fails to open some of the TIF images
    # But imageio doesn't have any problems
    return Image.fromarray(imageio.imread(filepath))


def compress_single_pil(im, filename, qf=None, qtable=None):
    assert im.mode == "L"

    if qtable is None:
        im.save(filename, quality=qf)
    elif qf is None:
        assert isinstance(qtable, np.ndarray) and qtable.shape == (8, 8)
        # Pillow expects list of qtables, each of which must be a list with 64 items
        im.save(filename, qtables=[qtable.flatten().tolist()])
        # Temporarily double-check qtable
        if not verify_qtable(filename, qtable):
            raise RuntimeError("Failed to write desired qtable")
    else:
        raise ValueError("Either quality factor `qf` or quantization table `qtable` must be given")


def compress_single_cjpeg(img, filename, qf=None, qtable=None, cjpeg_args=()):
    assert img.dtype == np.uint8

    if qtable is None:
        img_cjpeg(img, filename, quality=qf, cjpeg_args=cjpeg_args)
    elif qf is None:
        assert isinstance(qtable, np.ndarray) and qtable.shape == (8, 8)
        # Dump quantization table to text file
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            np.savetxt(f.name, qtable, fmt="%u", delimiter=" ", header="Custom luma quantization table")

            img_cjpeg(img, filename, quality=None, cjpeg_args=("-qtables", f.name) + cjpeg_args)
    else:
        raise ValueError("Either quality factor `qf` or quantization table `qtable` must be given")


def compress_double_pil(im, filename, qf1=None, qf2=None, qtable1=None, qtable2=None):
    # Verify args
    assert im.mode == "L"
    assert (qf1 is not None) ^ (qtable1 is not None), "Either `qf1` or `qtable1` must be given"
    assert (qf2 is not None) ^ (qtable2 is not None), "Either `qf2` or `qtable2` must be given"

    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        # First compression
        compress_single_pil(im, f.name, qf=qf1, qtable=qtable1)

        # Read back image content
        im = Image.open(f.name)
        assert im.mode == "L"

    # Second compression
    compress_single_pil(im, filename, qf=qf2, qtable=qtable2)


def compress_double_cjpeg(img, filename, qf1=None, qf2=None, qtable1=None, qtable2=None, cjpeg1_args=(), cjpeg2_args=()):
    assert img.dtype == np.uint8
    assert (qf1 is not None) ^ (qtable1 is not None), "Either `qf1` or `qtable1` must be given"
    assert (qf2 is not None) ^ (qtable2 is not None), "Either `qf2` or `qtable2` must be given"

    with tempfile.NamedTemporaryFile(suffix=".jpg") as sc_file:
        # First compression
        compress_single_cjpeg(img, sc_file.name, qf=qf1, qtable=qtable1, cjpeg_args=cjpeg1_args)

        # Second compression
        if qtable2 is None:
            djpeg_cjpeg(sc_file.name, filename, quality=qf2, cjpeg_args=cjpeg2_args)

        elif qf2 is None:
            assert isinstance(qtable2, np.ndarray) and qtable2.shape == (8, 8)
            # Dump quantization table to text file
            with tempfile.NamedTemporaryFile(suffix=".txt") as qtable2_file:
                np.savetxt(qtable2_file.name, qtable2, fmt="%u", delimiter=" ", header="Custom luma quantization table")

                # Pipe output of djpeg into cjpeg
                djpeg_cjpeg(sc_file.name, filename, quality=None, cjpeg_args=("-qtables", qtable2_file.name) + cjpeg2_args)

        else:
            raise ValueError("Either quality factor `qf2` or quantization table `qtable2` must be given")


def process_file_pil(filename, qf1=None, qf2=None, qtable1=None, qtable2=None, image_root=None, freq_bands=FREQ_BANDS, crops=None):
    filepath = filename
    if image_root is not None:
        filepath = os.path.join(image_root, filename)

    # Convert image to 8-bit grayscale
    im = open_tif(filepath).convert("L")

    # Single compression
    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        compress_single_pil(im, f.name, qf=qf2, qtable=qtable2)
        sc_features = extract_features_from_file(f.name, freq_bands=freq_bands, crops=crops)

    # Double compression
    with tempfile.NamedTemporaryFile(suffix=".jpg") as g:
        compress_double_pil(im, g.name, qf1=qf1, qf2=qf2, qtable1=qtable1, qtable2=qtable2)
        dc_features = extract_features_from_file(g.name, freq_bands=freq_bands, crops=crops)

    return filename, sc_features, dc_features


def process_file_cjpeg(filename, qf1=None, qf2=None, qtable1=None, qtable2=None, image_root=None, cjpeg_args=(), freq_bands=FREQ_BANDS, crops=None):
    filepath = filename
    if image_root is not None:
        filepath = os.path.join(image_root, filename)

    # Ensure to use same color2rgb conversion as PIL
    im = open_tif(filepath).convert("L")
    img = np.array(im)

    # Single compression
    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        compress_single_cjpeg(img, f.name, qf=qf2, qtable=qtable2, cjpeg_args=cjpeg_args)
        sc_features = extract_features_from_file(f.name, freq_bands=freq_bands, crops=crops)

    # Double compression
    with tempfile.NamedTemporaryFile(suffix=".jpg") as g:
        compress_double_cjpeg(img, g.name, qf1=qf1, qf2=qf2, qtable1=qtable1, qtable2=qtable2, cjpeg1_args=cjpeg_args, cjpeg2_args=cjpeg_args)
        dc_features = extract_features_from_file(g.name, freq_bands=freq_bands, crops=crops)

    return filename, sc_features, dc_features


def create_dataset(filenames, output_file, qf1s, qf2s, crops, image_root=None):
    combinations = list(itertools.product(qf1s, qf2s))

    # Make sure to remove all duplicates
    combinations = list(dict.fromkeys(combinations))

    with h5py.File(output_file, "w") as f:

        num_threads = mp.cpu_count()
        with mp.Pool(processes=num_threads) as p:
            for qf1, qf2 in tqdm(combinations, desc="Looping qf1-qf2 pairs"):
                # imap() preserves order but can cause significant drop in performance.
                # Change `process_file_pil` to `process_file_cjpeg` if you want to save JPEG images with custom settings.
                results = list(tqdm(p.imap(functools.partial(process_file_pil, qf1=qf1, qf2=qf2, image_root=image_root, crops=crops), filenames), desc="Feature extraction", total=len(filenames)))

                # Concatenate results to ndarrays
                all_filenames = []
                all_sc_features = []
                all_dc_features = []

                # Sequentially write results to file
                for (filename, sc_features, dc_features) in results:
                    all_filenames.append(filename)
                    all_sc_features.append(sc_features)
                    all_dc_features.append(dc_features)

                all_sc_features = np.stack(all_sc_features, axis=0)
                all_dc_features = np.stack(all_dc_features, axis=0)

                # Create one group for each (qf1, qf2) pair
                qf_pair_group = f.create_group(name="qf1_{:d}_qf2_{:d}".format(qf1, qf2))

                # Create a subgroup for each crop size
                for crop_idx, crop in enumerate(crops):
                    crop_group = qf_pair_group.create_group(name=str(crop))

                    # Create dataset with same shape as sc_features but crops are placed in different subgroups
                    shape = (all_sc_features.shape[0],) + all_sc_features.shape[2:]
                    sc_dset = crop_group.create_dataset(name="sc_features", shape=shape, dtype=np.float)
                    sc_dset[:] = all_sc_features[:, crop_idx]

                    dc_dset = crop_group.create_dataset(name="dc_features", shape=shape, dtype=np.float)
                    dc_dset[:] = all_dc_features[:, crop_idx]

                    shape = (len(all_filenames),)
                    filename_dset = crop_group.create_dataset(name="filename", shape=shape, dtype=h5py.special_dtype(vlen=str))
                    filename_dset[:] = all_filenames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, required=True, help="Path to TIF images")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store resulting HDF5 file(s)")
    args = vars(parser.parse_args())

    qfs = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    crops = [32, 64, 128, 256, FULL_RESOLUTION]

    image_root = args["image_root"]
    filenames = list(sorted(glob(os.path.join(image_root, "*.TIF"))))
    filenames = list(map(lambda f: os.path.relpath(f, image_root), filenames))

    output_filename = time.strftime("%Y_%m_%d") + "-benfords_law_features.h5"
    output_file = os.path.join(args["output_dir"], output_filename)
    assert not os.path.exists(output_file), "Output file \"{}\" exists already".format(output_file)

    create_dataset(
        filenames=filenames,
        output_file=output_file,
        qf1s=qfs,
        qf2s=qfs,
        crops=crops,
        image_root=image_root)
