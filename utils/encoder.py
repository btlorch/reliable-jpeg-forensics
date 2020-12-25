import tempfile
import subprocess
import imageio
from utils.constants import constants, LIBJPEG_CJPEG_EXECUTABLE_KEY, LIBJPEG_DJPEG_EXECUTABLE_KEY
import collections


def _is_tuple_or_list(args):
    if isinstance(args, str):
        return False

    return isinstance(args, collections.abc.Sequence)


def img_cjpeg(img, output_filename, quality, cjpeg_args=()):
    """
    Saves a given ndarray as JPEG image
    :param img: ndarray
    :param output_filename: path to output JPEG file
    :param quality: JPEG quality factor
    :param cjpeg_args: additional command line arguments to pass on to cjpeg
    """
    with tempfile.NamedTemporaryFile(suffix=".ppm") as f:
        imageio.imwrite(f.name, img)

        cjpeg(f.name, output_filename, quality, cjpeg_args)


def cjpeg(input_filename, output_filename, quality, cjpeg_args=()):
    """
    Compress a given image to JPEG using libjpeg's cjpeg executable.
    :param input_filename: path to input image
    :param output_filename: where to store resulting JPEG image
    :param quality: JPEG quality
    :param cjpeg_args: tuple/list of additional parameters passed to cjpeg
    :return: output filename
    """
    # Ensure cjpeg_args to be collections
    if not _is_tuple_or_list(cjpeg_args):
        raise ValueError("Additional arguments to cjpeg must be a list or a tuple")

    # Concatenate cjpeg command line
    cjpeg_bin = constants[LIBJPEG_CJPEG_EXECUTABLE_KEY]
    # Skip quality if qtables is set
    if "-qtables" not in cjpeg_args:
        cjpeg_command_line = [cjpeg_bin, "-quality", str(quality)]
    else:
        cjpeg_command_line = [cjpeg_bin]

    cjpeg_command_line = cjpeg_command_line + ["-outfile", output_filename, input_filename]

    if len(cjpeg_args) > 0:
        # Insert at position 1
        cjpeg_command_line[1:1] = list(cjpeg_args)

    # Raise error if exit code is non-zero
    cjpeg_process = subprocess.run(cjpeg_command_line, stdout=subprocess.PIPE, check=True)

    return output_filename


def djpeg_cjpeg(input_filename, output_filename, quality, djpeg_args=(), cjpeg_args=()):
    """
    Recompresses a given JPEG image
    :param input_filename: file path to image to be recompressed
    :param output_filename: file path where to store recompressed image
    :param quality: quality factor for cjpeg
    :param djpeg_args: tuple/list of additional arguments for djpeg command
    :param cjpeg_args: tuple/list of additional arguments for cjpeg
    :return: output raw_filename
    """
    # Ensure djpeg_args and cjpeg_args to be collections
    if not _is_tuple_or_list(djpeg_args) or not _is_tuple_or_list(cjpeg_args):
        raise ValueError("Additional arguments to djpeg and cjpeg must be a list or a tuple")

    djpeg_bin = constants[LIBJPEG_DJPEG_EXECUTABLE_KEY]
    cjpeg_bin = constants[LIBJPEG_CJPEG_EXECUTABLE_KEY]

    # Djpeg command line
    djpeg_command_line = [djpeg_bin, input_filename]
    if len(djpeg_args) > 0:
        # Insert at position 1
        djpeg_command_line[1:1] = list(djpeg_args)

    # Skip quality if qtables is set
    if "-qtables" not in cjpeg_args:
        cjpeg_command_line = [cjpeg_bin, "-quality", str(quality)]
    else:
        cjpeg_command_line = [cjpeg_bin]

    cjpeg_command_line = cjpeg_command_line + ["-outfile", output_filename]

    if len(cjpeg_args) > 0:
        # Insert at position 1
        cjpeg_command_line[1:1] = list(cjpeg_args)

    # Pipe output from djpeg directly into cjpeg
    djpeg_process = subprocess.Popen(djpeg_command_line, stdout=subprocess.PIPE)
    cjpeg_process = subprocess.run(cjpeg_command_line, stdin=djpeg_process.stdout, stdout=subprocess.PIPE, check=True)

    return output_filename


if __name__ == "__main__":
    import numpy as np

    img = np.random.randint(0, 256, size=(32, 32)).astype(np.uint8)
    img_cjpeg(img, "/tmp/bla.jpg", quality=85)
