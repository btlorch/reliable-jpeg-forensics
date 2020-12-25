from detector.variational_logistic_regression_detector import VariationalLogisticRegressionIsotropicPriorOutOfDistributionDetector, VariationalLogisticRegressionFullCovariancePriorOutOfDistributionDetector
from detector.k_nearest_neighbor_detector import KNearestNeighborOutOfDistributionDetector


VARIATIONAL_LOGISTIC_REGRESSION_ISOTROPIC_PRIOR = "variational_logistic_regression_isotropic_prior"
VARIATIONAL_LOGISTIC_REGRESSION_AXIS_ALIGNED_PRIOR = "variational_logistic_regression_axis_aligned_prior"
VARIATIONAL_LOGISTIC_REGRESSION_FULL_COVARIANCE_PRIOR = "variational_logistic_regression_full_covariance_prior"
K_NEAREST_NEIGHBORS = "k_nearest_neighbors"


def set_up_detector(detector_name, **detector_args):
    if VARIATIONAL_LOGISTIC_REGRESSION_ISOTROPIC_PRIOR == detector_name:
        return VariationalLogisticRegressionIsotropicPriorOutOfDistributionDetector(
            alpha=detector_args["alpha"],
            scale_mean=detector_args.get("scale_mean", True),
            scale_std=detector_args.get("scale_std", False))

    elif VARIATIONAL_LOGISTIC_REGRESSION_FULL_COVARIANCE_PRIOR == detector_name:
        return VariationalLogisticRegressionFullCovariancePriorOutOfDistributionDetector(
            prior_scale=detector_args["prior_scale"],
            scale_mean=detector_args.get("scale_mean", True),
            scale_std=detector_args.get("scale_std", False),
        )

    elif K_NEAREST_NEIGHBORS == detector_name:
        return KNearestNeighborOutOfDistributionDetector(
            n_neighbors=detector_args["n_neighbors"],
            scale_mean=detector_args.get("scale_mean", True),
            scale_std=detector_args.get("scale_std", False))

    elif callable(detector_name):
        return detector_name(**detector_args)

    else:
        raise ValueError("Unknown detector name")
