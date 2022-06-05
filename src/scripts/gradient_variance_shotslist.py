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


def var_shotslist():

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fd", dest="fd", action="store_false")
    parser.add_argument("--no-bd", dest="bd", action="store_false")
    parser.add_argument("--no-cd", dest="cd", action="store_false")
    parser.add_argument("--no-ps", dest="ps", action="store_false")
    parser.add_argument("--no-gps", dest="gps", action="store_false")
    parser.add_argument("--no-spsa", dest="spsa", action="store_false")

    parser.add_argument("--fd_h", type=float, default=0.01)
    parser.add_argument("--bd_h", type=float, default=0.01)
    parser.add_argument("--cd_h", type=float, default=0.01)
    parser.add_argument("--gps_gamma", type=float, default=0.5)
    parser.add_argument("--spsa_h", type=float, default=0.01)
    parser.add_argument("--spsa_count", type=int, default=10)

    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--shots_start", type=int, default=10)
    parser.add_argument("--shots_stop", type=int, default=101)
    parser.add_argument("--shots_step", type=int, default=10)

    parser.add_argument("--ansatz", type=str, default="ESU2")
    parser.add_argument("--observable", type=str, default="Diagonal")
    parser.add_argument("--metric", type=str, default="MSE")

    args = parser.parse_args()

    fd_flag = args.fd
    bd_flag = args.bd
    cd_flag = args.cd
    gps_flag = args.gps
    spsa_flag = args.spsa
    ps_flag = args.ps

    fd_h = args.fd_h
    bd_h = args.bd_h
    cd_h = args.cd_h
    gps_gamma = args.gps_gamma
    spsa_h = args.spsa_h
    spsa_count = args.spsa_count

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
    fd = GradientSampleEstimator(
        FiniteDifferenzen(ansatz, observable, [0, fd_h], "opflow", label=f"FD {fd_h}"),
        1,
    )
    bd = GradientSampleEstimator(
        FiniteDifferenzen(ansatz, observable, [0, -bd_h], "opflow", label=f"BD {bd_h}"),
        1,
    )
    cd = GradientSampleEstimator(
        FiniteDifferenzen(
            ansatz, observable, [-cd_h, cd_h], "opflow", label=f"CD {cd_h}"
        ),
        1,
    )
    ps = GradientSampleEstimator(ParameterShift(ansatz, observable, "opflow"), 1)
    gps = GradientSampleEstimator(
        GeneralParameterShift(ansatz, observable, gps_gamma, "opflow"), 1
    )
    spsa = GradientSampleEstimator(
        SPSAGradient(ansatz, observable, spsa_h, "opflow"), spsa_count
    )

    if fd_flag:
        gradient_list.append(fd)
    if bd_flag:
        gradient_list.append(bd)
    if cd_flag:
        gradient_list.append(cd)
    if ps_flag:
        gradient_list.append(ps)
    if gps_flag:
        gradient_list.append(gps)
    if spsa_flag:
        gradient_list.append(spsa)

    point = generate_random_rotation_point(ansatz.get_parameter_count())
    parameter_indices = range(ansatz.get_parameter_count())

    shots_list = range(shots_start, shots_stop, shots_step)
    compare_variance_bias_shotslist(
        gradient_list, point, shots_list, parameter_indices, std_dev
    )


if __name__ == "__main__":
    var_shotslist()
