import numpy as np
from collections.abc import Callable

# different error metrics


def mean_squared_error(sample: np.ndarray, exact: np.ndarray) -> np.float:

    assert sample.shape == exact.shape

    error = np.average(np.power(sample - exact, 2))
    return error.real


def squared_error(sample: np.ndarray, exact: np.ndarray) -> np.ndarray:

    assert sample.shape == exact.shape

    error = np.power(sample - exact, 2)
    return error.real


def mean_absolute_error(sample: np.ndarray, exact: np.ndarray) -> np.float:

    assert sample.shape == exact.shape

    error = np.average(np.abs(sample - exact))
    return error.real


def absolute_error(sample: np.ndarray, exact: np.ndarray) -> np.ndarray:

    assert sample.shape == exact.shape

    error = np.abs(sample - exact)
    return error.real


def get_metric_name(f: Callable, abbreviation: bool = False) -> str:

    # get metric name for plots

    if f.__name__ == "mean_squared_error":
        if abbreviation:
            return "MSE"
        else:
            # return "Mean Squared Error"
            return "Mittlere Quadratische Abweichung"

    elif f.__name__ == "squared_error":
        if abbreviation:
            return "SE"
        else:
            return "Quadratische Abweichung"

    elif f.__name__ == "mean_absolute_error":
        if abbreviation:
            return "MAE"
        else:
            return "Mittlere Absolute Abweichung"

    elif f.__name__ == "absolute_error":
        if abbreviation:
            return "AE"
        else:
            return "Absolute Abweichung"

    else:
        return "Unknown Metric"
