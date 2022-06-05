import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.IAGQ.experiment import (
    compare_gradients_shotslist,
    compare_variance_bias_shotslist,
)
from src.IAGQ.gradient import (
    SPSAGradient,
    ParameterShift,
    GeneralParameterShift,
    FiniteDifferenzen,
)
from src.IAGQ.tools import GradientSampleEstimator, generate_random_rotation_point

import argparse
import numpy as np
from src.IAGQ.ansatz import (
    CR_Euler_Ansatz,
    EfficientSU2Wrapper,
    LayeredHardwareEfficientAnsatz,
)
from src.IAGQ.observable import (
    DiagonalObservable,
    VQE_LiH_Observable,
    VQE_HeHObservable,
    VQE_H2_Observable,
)
from src.IAGQ.metrics import mean_absolute_error, mean_squared_error


def var_gps_shotslist():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-g",
        "--gamma_list",
        nargs="+",
        help="<Required> Set flag",
        required=True,
        type=float,
    )

    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--shots_start", type=int, default=10)
    parser.add_argument("--shots_stop", type=int, default=101)
    parser.add_argument("--shots_step", type=int, default=10)

    parser.add_argument("--ansatz", type=str, default="ESU2")
    parser.add_argument("--observable", type=str, default="Diagonal")
    parser.add_argument("--metric", type=str, default="MSE")

    args = parser.parse_args()

    gamma_list = args.gamma_list

    seed = args.seed
    shots_start = args.shots_start
    shots_stop = args.shots_stop
    shots_step = args.shots_step

    ansatz_string = args.ansatz
    observable_string = args.observable
    metric_string = args.metric

    if observable_string == "Diagonal":
        observable = DiagonalObservable(4, weights=[2 * i for i in range(2**4)])
    elif observable_string == "LiH":
        observable = VQE_LiH_Observable()
    elif observable_string == "HeH":
        observable = VQE_HeHObservable(r=0.1)
    elif observable_string == "H2":
        observable = VQE_H2_Observable()
    else:
        raise Exception("Unknown observable")

    if ansatz_string == "ESU2":
        ansatz = EfficientSU2Wrapper(observable.qubits, "linear", 2)
    elif ansatz_string == "CRE":
        ansatz = CR_Euler_Ansatz(observable.qubits, 2)
    elif ansatz_string == "LHE":
        ansatz = LayeredHardwareEfficientAnsatz(observable.qubits, 2)
    else:
        raise Exception("Unknown ansatz")

    if metric_string == "MSE":
        std_dev = False
    elif metric_string == "MAE":
        std_dev = True
    else:
        raise Exception("Unknown metric")

    np.random.seed(seed)
    gradient_list = []

    ps = GradientSampleEstimator(ParameterShift(ansatz, observable, "opflow"), 1)
    gradient_list.append(ps)
    for gamma in gamma_list:

        gps = GradientSampleEstimator(
            GeneralParameterShift(
                ansatz, observable, gamma, "opflow", label=f"GPS {gamma}"
            ),
            1,
        )
        gradient_list.append(gps)

    point = generate_random_rotation_point(ansatz.get_parameter_count())
    parameter_indices = range(ansatz.get_parameter_count())

    shots_list = range(shots_start, shots_stop, shots_step)
    compare_variance_bias_shotslist(
        gradient_list, point, shots_list, parameter_indices, std_dev
    )


if __name__ == "__main__":
    var_gps_shotslist()
