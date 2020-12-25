from utils.logger import setup_custom_logger
import numpy as np
import itertools
import h5py
import os


log = setup_custom_logger(os.path.basename(os.path.basename(__file__)))


class BufferedWriter(object):
    def __init__(self, output_filename, chunk_size=256, required_keys=()):
        self._output_filename = output_filename
        self._chunk_size = chunk_size
        self._required_keys = required_keys

        # Buffering data structures for incremental writes
        self._written_items_count = 0
        # Set up empty list for each buffer
        self._wait_for_write_buffer = dict()
        self._num_samples_in_write_buffer = 0
        self._chunk_remainder_buffer = dict()
        self._num_samples_in_chunk_remainder_buffer = 0

    @staticmethod
    def _count_samples(buffers_dict):
        """
        Counts the number of samples in the given dictionary.
        Assumes that the number of samples is the same for each data set.
        Empty data sets are ignored.
        :param buffers_dict: dictionary with different data sets
        :return: 0 if dictionary is empty or the number of samples
        """
        if not buffers_dict:
            return 0

        for key in list(buffers_dict.keys()):
            num_items = len(buffers_dict[key])
            # There might be empty data sets in the buffer which we want to ignore
            if 0 != num_items:
                return num_items

        return 0

    def _concat_buffers(self):
        """
        Concatenates the items in the wait_for_write_buffer itself.
        Concatenates the result with the remaining items in the _chunk_remainder.buffer.
        :return: Dictionary of data sets, where each value is either a flat list or an ndarray with the number of samples in the first dimension
        """
        if self._num_samples_in_write_buffer == 0:
            return self._chunk_remainder_buffer

        # Concatenate samples from list
        concat_buffer = dict()
        for key, data_buffer in self._wait_for_write_buffer.items():
            # Ignore empty data buffers
            if len(data_buffer) == 0:
                continue

            # Data buffer contains a list of batches to concatenate
            if isinstance(data_buffer[0], list):
                concat_buffer[key] = list(itertools.chain.from_iterable(data_buffer))
            elif isinstance(data_buffer[0], np.ndarray):
                concat_buffer[key] = np.concatenate(data_buffer)
            else:
                log.error("Unexpected buffer type")
                raise ValueError("Unexpected buffer type")

        # Concatenate with remaining data from previous write
        if self._num_samples_in_chunk_remainder_buffer > 0:
            if set(concat_buffer.keys()) != set(self._chunk_remainder_buffer.keys()):
                log.error("Keys differ across different batches. Make sure you provide the same set of keys for each batch.")
                raise ValueError("Keys differ across different batches. Make sure you provide the same set of keys for each batch.")

            for key, data_buffer in concat_buffer.items():
                if isinstance(data_buffer, list):
                    concat_buffer[key] = self._chunk_remainder_buffer[key] + data_buffer
                elif isinstance(data_buffer, np.ndarray):
                    concat_buffer[key] = np.concatenate((self._chunk_remainder_buffer[key], data_buffer))
                else:
                    log.error("Unexpected buffer type")
                    raise ValueError("Unexpected buffer type")

        return concat_buffer

    def write(self, batch):
        """
        Writes the given batch to a local buffer.
        Once the local buffer reaches the predefined chunk size, multiples of the chunk size are written to disk.
        The remaining items are kep in the chunk_remainder_buffer.
        :param batch: dictionary with data sets as list
        :return:
        """

        num_batch_samples = self._count_samples(batch)

        # Make sure that the batch contains at least all required keys
        for key in self._required_keys:
            assert key in batch, "Given batch does not contain key {}".format(key)

        # Copy batch data sets into local buffers
        for key, data_buffer in batch.items():
            # Create new list if key was not seen before
            if key not in self._wait_for_write_buffer:
                self._wait_for_write_buffer[key] = []

            # Append to existing list
            self._wait_for_write_buffer[key].append(data_buffer)

        self._num_samples_in_write_buffer += num_batch_samples

        total_num_samples = self._num_samples_in_write_buffer + self._num_samples_in_chunk_remainder_buffer
        # Once we have reached to chunk size, write as many multiples of the chunk size as possible to disk.
        if total_num_samples >= self._chunk_size:
            concat_buffer = self._concat_buffers()

            while total_num_samples >= self._chunk_size:
                # Extract a chunk of data
                chunk = dict()
                for key, data_buffer in concat_buffer.items():
                    # Move chunk into new buffer
                    chunk[key] = data_buffer[:self._chunk_size]
                    # Remove chunk from buffer
                    concat_buffer[key] = data_buffer[self._chunk_size:]

                self._write(chunk)
                total_num_samples -= self._chunk_size

            self._chunk_remainder_buffer = concat_buffer
            # Reset write buffer
            for buffer in self._wait_for_write_buffer.values():
                buffer.clear()

            self._num_samples_in_write_buffer = 0
            self._num_samples_in_chunk_remainder_buffer = self._count_samples(self._chunk_remainder_buffer)

    def _write(self, chunk):
        """

        :param chunk: data already concatenated and ready to be written
        :return:
        """
        chunk_size = self._count_samples(chunk)
        if 0 == chunk_size:
            return

        if 0 == self._written_items_count:
            # Create new output file
            with h5py.File(self._output_filename, "w") as f:
                # Initialize a resizable data set to hold the output
                for key, data_buffer in chunk.items():
                    kwargs = {}
                    if isinstance(data_buffer, list):
                        shape = (chunk_size,)
                        maxshape = (None,)
                        # Select dtype based on data's type
                        if isinstance(data_buffer[0], (np.integer, int)):
                            dtype = int
                        elif isinstance(data_buffer[0], (np.floating, float)):
                            dtype = float
                        elif isinstance(data_buffer[0], str):
                            dtype = h5py.special_dtype(vlen=str)
                        else:
                            dtype = h5py.special_dtype(vlen=bytes)

                    elif isinstance(data_buffer, np.ndarray):
                        shape = data_buffer.shape
                        maxshape = (None,) + data_buffer.shape[1:]
                        dtype = data_buffer.dtype
                        kwargs["compression"] = "gzip"
                        kwargs["chunks"] = (self._chunk_size,) + data_buffer.shape[1:]
                    else:
                        log.error("Unknown item type")

                    # Set up data set
                    dataset = f.create_dataset(key, shape=shape, maxshape=maxshape, dtype=dtype, **kwargs)
                    dataset[:] = data_buffer

        else:
            # Append to existing output file
            with h5py.File(self._output_filename, "a") as f:
                for key, data_buffer in chunk.items():
                    dataset = f[key]
                    # Resize the data set to accommodate the next chunk of rows
                    dataset.resize(self._written_items_count + chunk_size, axis=0)
                    # Write the next chunk
                    dataset[self._written_items_count:] = data_buffer

        # Increment the row counter
        self._written_items_count += chunk_size

    def flush(self):
        if 0 == self._num_samples_in_write_buffer + self._num_samples_in_chunk_remainder_buffer:
            return

        chunk = self._concat_buffers()
        self._write(chunk)
        self._num_samples_in_write_buffer = 0
        self._num_samples_in_chunk_remainder_buffer = 0
