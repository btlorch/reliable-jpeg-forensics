from detector.bootstrap import VARIATIONAL_LOGISTIC_REGRESSION_FULL_COVARIANCE_PRIOR
from experiments.eval_qf2_mismatch_iso_prior import loop
from utils.logger import setup_custom_logger
from tqdm.auto import tqdm
import pandas as pd
import argparse
import time
import os


log = setup_custom_logger(os.path.basename(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required args
    parser.add_argument("--prior_scale", required=True, type=float, help="Scale factor for covariance-based prior", nargs="+")
    parser.add_argument("--data_filename", required=True, type=str, help="Filename of data file")
    parser.add_argument("--data_dir", required=True, type=str, help="Directory where HDF5 datasets are located")

    # Optional args
    parser.add_argument("--num_repeats", type=int, help="How many detectors to train on random in-distribution-data splits", default=10)
    parser.add_argument("--output_dir", type=str, help="Directory where to store results. By default uses the given `data_dir`", default=None)
    parser.add_argument("--no_scale_mean", dest="scale_mean", action="store_false")
    parser.set_defaults(scale_mean=True)
    args = vars(parser.parse_args())

    results_csv = time.strftime("%Y_%m_%d") + "-eval_qf2_mismatch_single_positive_class_full_covariance_prior.csv"
    buffer = []
    for prior_scale in tqdm(args["prior_scale"], desc="Iterating prior scales"):
        df = loop(detector_name=VARIATIONAL_LOGISTIC_REGRESSION_FULL_COVARIANCE_PRIOR,
                  detector_args={
                      "prior_scale": prior_scale,
                      "scale_mean": args["scale_mean"],
                  },
                  num_repeats=args["num_repeats"],
                  data_file=os.path.join(args["data_dir"], args["data_filename"]))

        buffer.append(df)

    df = pd.concat(buffer)

    output_dir = args["output_dir"] if args["output_dir"] is not None else args["data_dir"]
    results_filepath = os.path.join(output_dir, results_csv)

    df.to_csv(results_filepath, index=False)
    log.info("Written results to \"{}\"".format(results_filepath))
