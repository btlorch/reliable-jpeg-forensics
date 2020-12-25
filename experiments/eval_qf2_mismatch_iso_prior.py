from detector.bootstrap import set_up_detector, VARIATIONAL_LOGISTIC_REGRESSION_ISOTROPIC_PRIOR
from data.benfords_law_features import FULL_RESOLUTION
from data.data_controller_v2 import DataControllerV2
from utils.logger import setup_custom_logger
from data.split import FilenameShuffleSplit
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import itertools
import argparse
import time
import os


log = setup_custom_logger(os.path.basename(__file__))


def eval_qf2_mismatch(num_repeats, data_controller, qf1s, qf2_train, qf2s_test, detector_name, detector_args, crop, verbose=False):
    """
    :param num_repeats: number of times to repeat the experiment
    :param data_controller: instance of DataControllerV2
    :param qf1s: first compression quality factors to be included for the positive class. Can be scalar or list of scalars
    :param qf2_train: second compression quality used for training
    :param qf2s_test: list of second compression qualities on which to test the trained detector on
    :param detector_name: name of out-of-distribution detector
    :param detector_args: parameters passed to out-of-distribution detector constructor
    :param crop: retrieve features of the given crop size, either number of FULL_RESOLUTION constant
    :param verbose: pass verbose flag to detector's fit() method
    :return: list of dicts. Each list items carries the result of one qf2_test
    """
    buffer = []

    # Keep a list of detectors
    detectors = []

    # Make sure to set random state, then we can re-iterate over the same splits again for evaluation
    splitter = FilenameShuffleSplit(n_splits=num_repeats, test_size=0.5, random_state=91058)
    img_filenames = data_controller.get_filenames()
    for train_filenames, test_filenames in splitter.split(img_filenames):
        X_train, y_train, X_test, y_test = data_controller.get_data(qf1s=qf1s, qf2s=qf2_train, train_filenames=train_filenames, test_filenames=test_filenames, crop=crop)

        # Train detector
        detector = set_up_detector(detector_name=detector_name, **detector_args)
        detector.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, verbose=verbose)
        detectors.append(detector)

    # Evaluate for different out-of-distribution qf2
    for qf2_test in qf2s_test:
        for i, (train_filenames, test_filenames) in enumerate(splitter.split(img_filenames)):
            X_train, y_train, X_test, y_test = data_controller.get_data(qf1s=qf1s, qf2s=qf2_test, train_filenames=train_filenames, test_filenames=test_filenames, crop=crop)
            detector = detectors[i]

            res = {
                "qf1": qf1s,
                "qf2_train": qf2_train,
                "qf2_test": qf2_test,
                "ind_accuracy": detector.in_distribution_accuracy(),
                "sc_dc_distance": detector.inter_class_distance,
                "crop": crop,
            }

            res.update(detector.eval_ood(X_test, y_test))
            res.update(detector.additional_scores())

            buffer.append(res)

    return buffer


def loop(num_repeats,
         data_file,
         detector_name,
         detector_args,
         qf1s=[50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
         qf2s=[50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
         crops=[32, 64, 128, 256, FULL_RESOLUTION],
         verbose=False):

    data_controller = DataControllerV2(data_file=data_file, use_cache=True)

    buffer = []
    combinations = list(itertools.product(qf1s, qf2s, crops))

    for qf1, qf2_train, crop in tqdm(combinations, desc="Iterating qf1, qf2_train, and crops"):
        qf2_train_buffer = eval_qf2_mismatch(num_repeats=num_repeats,
                                             data_controller=data_controller,
                                             qf1s=qf1,
                                             qf2_train=qf2_train,
                                             qf2s_test=qf2s,
                                             detector_name=detector_name,
                                             detector_args=detector_args,
                                             crop=crop,
                                             verbose=verbose)
        buffer.extend(qf2_train_buffer)

    return pd.DataFrame(buffer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required args
    parser.add_argument("--alpha", required=True, type=float, help="Precision of prior", nargs="+")
    parser.add_argument("--data_filename", required=True, type=str, help="Filename of data file")
    parser.add_argument("--data_dir", required=True, type=str, help="Directory where HDF5 datasets are located")

    # Optional args
    parser.add_argument("--num_repeats", type=int, help="How many detectors to train on random in-distribution-data splits", default=10)
    parser.add_argument("--output_dir", type=str, help="Directory where to store results. By default uses the given `data_dir`", default=None)
    parser.add_argument("--no_scale_mean", dest="scale_mean", action="store_false")
    parser.set_defaults(scale_mean=True)
    args = vars(parser.parse_args())

    results_csv = time.strftime("%Y_%m_%d") + "-eval_qf2_mismatch_single_positive_class_iso_prior.csv"
    buffer = []
    for alpha in tqdm(args["alpha"], desc="Iterating alphas"):
        df = loop(detector_name=VARIATIONAL_LOGISTIC_REGRESSION_ISOTROPIC_PRIOR,
                  detector_args={
                      "alpha": alpha,
                      "scale_mean": args["scale_mean"]
                  },
                  num_repeats=args["num_repeats"],
                  data_file=os.path.join(args["data_dir"], args["data_filename"]))

        buffer.append(df)

    df = pd.concat(buffer)

    output_dir = args["output_dir"] if args["output_dir"] is not None else args["data_dir"]
    results_filepath = os.path.join(output_dir, results_csv)

    df.to_csv(results_filepath, index=False)
    log.info("Written results to \"{}\"".format(results_filepath))
