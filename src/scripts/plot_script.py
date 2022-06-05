import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.IAGQ.ansatz import (
    LayeredHardwareEfficientAnsatz,
    CR_Euler_Ansatz,
    EfficientSU2Wrapper,
)
from src.IAGQ.observable import (
    DiagonalObservable,
    VQE_HeHObservable,
    VQE_H2_Observable,
    VQE_LiH_Observable,
)
import numpy as np
from src.IAGQ.gradient import (
    ParameterShift,
    FiniteDifferenzen,
    QGFGradient,
    SPSAGradient,
    GeneralParameterShift,
    Gradient,
)
import matplotlib.pyplot as plt
from src.IAGQ.experiment import (
    run_experiment,
    read_experiment_data,
    compare_gradients_shotslist,
    compare_variance_bias_shotslist,
    compare_sample_true_variance_shotslist,
    evaluate_variance_bias_shotslist,
    evaluate_metric_shotslist,
    compare_bias,
    evaluate_fd_bias_h_slist,
)
from src.IAGQ.tools import (
    GradientSampleEstimator,
    generate_random_rotation_point,
    one_param_rotation_grid,
    eq_superposition_variance,
)
from src.IAGQ.metrics import get_metric_name, mean_squared_error, mean_absolute_error
from cycler import cycler


def bias_plot():

    np.random.seed(40)
    ansatz = EfficientSU2Wrapper(qubits=4, layers=2, entanglement="linear")

    observable = DiagonalObservable(4, [2 * i for i in range(2**4)])

    point = generate_random_rotation_point(ansatz.get_parameter_count())
    shots_list = [100 * i for i in range(10, 110, 5)]
    parameter_indices = [i for i in range(ansatz.get_parameter_count())]
    metric = mean_squared_error

    gradient_list = []

    gradient_list.append(
        FiniteDifferenzen(
            ansatz,
            observable,
            [0, 0.01],
            "opflow",
            label="Forward Finite Differences h 0.01",
            plot_label="Vorwärts Finite Differenzen, h=0.01",
        ),
    )

    gradient_list.append(
        FiniteDifferenzen(
            ansatz,
            observable,
            [0, -0.01],
            "opflow",
            label="Backward Finite Differences h 0.01",
            plot_label="Rückwärts Finite Differenzen, h=0.01",
        ),
    )

    gradient_list.append(
        FiniteDifferenzen(
            ansatz,
            observable,
            [0.01, -0.01],
            "opflow",
            label="Central Finite Differences h 0.01",
            plot_label="Zentrale Finite Differenzen, h=0.01",
        ),
    )

    gradient_list.append(
        GradientSampleEstimator(
            SPSAGradient(
                ansatz,
                observable,
                0.01,
                "opflow",
                label="SPSA h 0.01",
                plot_label="SPSA, h=0.01",
            ),
            10,
            add_label=False,  # ansatz.get_parameter_count(),
        )
    )

    gradient_list.append(
        GeneralParameterShift(
            ansatz,
            observable,
            0.5,
            "opflow",
            label="General Parameter-Shift gamma 0.5",
            plot_label="Allgemeines Parameter-Shift, $\gamma$=0.5",
        ),
    )

    gradient_list.append(
        ParameterShift(
            ansatz,
            observable,
            "opflow",
            label="Simple Parameter-Shift",
            plot_label="Klassisches Parameter-Shift",
        )
    )

    # metric = mean_absolute_error

    compare_bias(
        gradient_list,
        point,
        parameter_indices,
        metric,
        "tab10",
        tab10_indices=range(len(gradient_list)),
    )


def gPS_PS_plot():

    np.random.seed(40)
    ansatz = EfficientSU2Wrapper(qubits=4, layers=2, entanglement="linear")

    observable = DiagonalObservable(4, [2 * i for i in range(2**4)])

    point = generate_random_rotation_point(ansatz.get_parameter_count())
    shots_list = [100 * i for i in range(10, 110, 5)]
    parameter_indices = [i for i in range(ansatz.get_parameter_count())]
    metric = mean_squared_error

    gradient_list = []

    gradient_list.append(
        GeneralParameterShift(
            ansatz,
            observable,
            0.5,
            "simulator",
            label="General Parameter-Shift gamma 0.5",
            plot_label="Allgemeines Parameter-Shift, $\gamma$=0.5",
        ),
    )
    gradient_list.append(
        ParameterShift(
            ansatz,
            observable,
            "simulator",
            label="Simple Parameter-Shift",
            plot_label="Klassisches Parameter-Shift",
        ),
    )

    compare_gradients_shotslist(
        gradient_list,
        point,
        shots_list,
        parameter_indices,
        metric,
        yscale="linear",
        cm="tab10",
        tab10_indices=[4, 5],
    )


def fd_konv_plot():

    np.random.seed(40)
    ansatz = EfficientSU2Wrapper(qubits=4, layers=2, entanglement="linear")

    observable = DiagonalObservable(4, [2 * i for i in range(2**4)])

    point = generate_random_rotation_point(ansatz.get_parameter_count())
    shots_list = [100 * i for i in range(15, 110, 5)]
    parameter_indices = [i for i in range(ansatz.get_parameter_count())]
    metric = mean_squared_error

    gradient_list = []

    # fd_vergleich plt
    gradient_list.append(
        FiniteDifferenzen(
            ansatz,
            observable,
            [0.01, -0.01],
            "simulator",
            label="Central Finite Differences h 0.01",
            plot_label="Zentrale Finite Differenzen, h=0.01",
        ),
    )

    gradient_list.append(
        FiniteDifferenzen(
            ansatz,
            observable,
            [0.005, -0.005],
            "simulator",
            label="Central Finite Differences h 0.005",
            plot_label="Zentrale Finite Differenzen, h=0.005",
        ),
    )

    gradient_list.append(
        FiniteDifferenzen(
            ansatz,
            observable,
            [0.001, -0.001],
            "simulator",
            label="Central Finite Differences h 0.001",
            plot_label="Zentrale Finite Differenzen, h=0.001",
        ),
    )

    compare_gradients_shotslist(
        gradient_list,
        point,
        shots_list,
        parameter_indices,
        metric,
        yscale="linear",
        cm="viridis",
    )


def fd_diff_h_plot():

    np.random.seed(40)
    ansatz = EfficientSU2Wrapper(qubits=4, layers=2, entanglement="linear")

    observable = DiagonalObservable(4, [2 * i for i in range(2**4)])

    point = generate_random_rotation_point(ansatz.get_parameter_count())
    shots_list = [100 * i for i in range(10, 110, 5)]
    parameter_indices = [i for i in range(ansatz.get_parameter_count())]
    metric = mean_squared_error

    gradient_list = []

    # fh_diff
    h_list = [
        [-0.5, 0.5],
        [-0.1, 0.1],
        [-0.5, -0.1, 0.1, 0.5],
        [-0.01, 0.01],
        [-0.1, -0.01, 0.01, 0.1],
        [-0.001, 0.001],
    ]

    for i in range(len(h_list)):
        h = h_list[i]
        if len(h) == 2:
            pl = f", h={h[1]}"
            l = f" {h[1]}"
        else:
            pl = f",$h_0$={h[2]}, $h_1$={h[3]}"
            l = f" {h[2]} {h[3]}"

        fd = FiniteDifferenzen(
            ansatz,
            observable,
            h,
            "simulator",
            label="Central Differences h" + l,
            plot_label="Zentrale Finite Differenzen" + pl,
        )
        gradient_list.append(fd)

    compare_gradients_shotslist(
        gradient_list, point, shots_list, parameter_indices, metric, cm="viridis"
    )


def fd_bias_h_plot():

    np.random.seed(40)
    ansatz = EfficientSU2Wrapper(qubits=4, layers=2, entanglement="linear")

    observable = DiagonalObservable(4, [2 * i for i in range(2**4)])

    point = generate_random_rotation_point(ansatz.get_parameter_count())
    parameter_indices = [i for i in range(ansatz.get_parameter_count())]
    metric = mean_squared_error

    gradient_list = []

    # bias_FD
    n = 20
    h_list = [10 ** ((-5 * i) / n) for i in range(n)]
    print(h_list)

    for h in h_list:
        gradient_list.append(
            FiniteDifferenzen(
                ansatz,
                observable,
                [-h, h],
                "opflow",
                label="ZFD",
            ),
        )

    evaluate_fd_bias_h_slist(gradient_list, point, h_list, parameter_indices, metric)


def gPS_gamma_plot():

    np.random.seed(40)
    ansatz = EfficientSU2Wrapper(qubits=4, layers=2, entanglement="linear")

    observable = DiagonalObservable(4, [2 * i for i in range(2**4)])

    point = generate_random_rotation_point(ansatz.get_parameter_count())
    shots_list = [100 * i for i in range(10, 110, 5)]
    parameter_indices = [i for i in range(ansatz.get_parameter_count())]
    metric = mean_squared_error

    gradient_list = []
    # var_vergleich_gPS
    g_list = [np.pi / 4, 0.5, 0.1, 0.01]
    gl = ["$\pi$/4", "0.5", "0.1", "0.01"]

    for i in range(len(g_list)):
        g = g_list[i]

        pl = f", $\gamma$=" + gl[i]
        gradient_list.append(
            GradientSampleEstimator(
                GeneralParameterShift(
                    ansatz,
                    observable,
                    g,
                    "opflow",
                    label=f"general Parameter-Shift g {g}",
                    plot_label="Allgemeines Parameter-Shift" + pl,
                ),
                1,
            )
        )
    compare_variance_bias_shotslist(
        gradient_list, point, shots_list, parameter_indices, std_dev=False, cm="autumn"
    )


def var_fd_plot():
    np.random.seed(40)
    ansatz = EfficientSU2Wrapper(qubits=4, layers=2, entanglement="linear")

    observable = DiagonalObservable(4, [2 * i for i in range(2**4)])

    point = generate_random_rotation_point(ansatz.get_parameter_count())
    shots_list = [100 * i for i in range(10, 110, 5)]
    parameter_indices = [i for i in range(ansatz.get_parameter_count())]
    metric = mean_squared_error
    gradient_list = []
    # var_vergleich_FD
    h_list = [0.5, 0.1, 0.01, 0.001]

    for i in range(len(h_list)):
        h = h_list[i]

        pl = f", h={h}"
        gradient_list.append(
            GradientSampleEstimator(
                FiniteDifferenzen(
                    ansatz,
                    observable,
                    [h, -h],
                    "opflow",
                    label=f"Central Differences h {h}",
                    plot_label="Zentrale Finite Differenzen" + pl,
                ),
                1,
            )
        )
    compare_variance_bias_shotslist(
        gradient_list, point, shots_list, parameter_indices, std_dev=False, cm="viridis"
    )


def var_vergleich_plot():
    np.random.seed(40)
    ansatz = EfficientSU2Wrapper(qubits=4, layers=2, entanglement="linear")

    observable = DiagonalObservable(4, [2 * i for i in range(2**4)])

    point = generate_random_rotation_point(ansatz.get_parameter_count())
    shots_list = [10 * i for i in range(10, 110, 5)]
    parameter_indices = [i for i in range(ansatz.get_parameter_count())]
    metric = mean_squared_error

    gradient_list = []

    # var plot
    bw = GradientSampleEstimator(
        FiniteDifferenzen(
            ansatz,
            observable,
            [0, 0.01],
            "opflow",
            label="Forward Finite Differences h 0.01",
            plot_label="Vorwärts Finite Differenzen, h=0.01",
        ),
        1,
    )
    gradient_list.append(bw)
    fw = GradientSampleEstimator(
        FiniteDifferenzen(
            ansatz,
            observable,
            [0, -0.01],
            "opflow",
            label="Backward Finite Differences h 0.01",
            plot_label="Rückwärts Finite Differenzen, h=0.01",
        ),
        1,
    )

    gradient_list.append(fw)
    gradient_list.append(
        GradientSampleEstimator(
            FiniteDifferenzen(
                ansatz,
                observable,
                [0.01, -0.01],
                "opflow",
                label="Central Finite Differences h 0.01",
                plot_label="Zentrale Finite Differenzen, h=0.01",
            ),
            1,
        )
    )
    gradient_list.append(
        GradientSampleEstimator(
            SPSAGradient(
                ansatz,
                observable,
                0.01,
                "opflow",
                label="SPSA h 0.01",
                plot_label="SPSA, h=0.01",
            ),
            10,  # ansatz.get_parameter_count(),
        )
    )

    gradient_list.append(
        GradientSampleEstimator(
            GeneralParameterShift(
                ansatz,
                observable,
                0.5,
                "opflow",
                label="General Parameter-Shift gamma 0.5",
                plot_label="Allgemeines Parameter-Shift, $\gamma$=0.5",
            ),
            1,
        )
    )

    gradient_list.append(
        GradientSampleEstimator(
            ParameterShift(
                ansatz,
                observable,
                "opflow",
                label="Simple Parameter-Shift",
                plot_label="Klassisches Parameter-Shift",
            ),
            1,
        )
    )
    compare_variance_bias_shotslist(
        gradient_list,
        point,
        shots_list,
        parameter_indices,
        std_dev=False,
        cm="tab10",
        tab10_indices=range(len(gradient_list)),
    )


def PS_gPS_plot():
    np.random.seed(40)
    ansatz = EfficientSU2Wrapper(qubits=4, layers=2, entanglement="linear")

    observable = DiagonalObservable(4, [2 * i for i in range(2**4)])

    point = generate_random_rotation_point(ansatz.get_parameter_count())
    shots_list = [100 * i for i in range(10, 110, 5)]
    parameter_indices = [i for i in range(ansatz.get_parameter_count())]
    metric = mean_squared_error

    gradient_list = []

    gradient_list.append(
        GeneralParameterShift(
            ansatz,
            observable,
            0.5,
            "simulator",
            label="General Parameter-Shift gamma 0.5",
            plot_label="Allgemeines Parameter-Shift, $\gamma$=0.5",
        ),
    )
    gradient_list.append(
        ParameterShift(
            ansatz,
            observable,
            "simulator",
            label="Simple Parameter-Shift",
            plot_label="Klassisches Parameter-Shift",
        ),
    )
    compare_gradients_shotslist(
        gradient_list,
        point,
        shots_list,
        parameter_indices,
        metric,
        cm="tab10",
        yscale="linear",
        tab10_indices=[4, 5],
    )


def fd_spsa_plot():
    np.random.seed(40)
    ansatz = EfficientSU2Wrapper(qubits=4, layers=2, entanglement="linear")

    observable = DiagonalObservable(4, [2 * i for i in range(2**4)])

    point = generate_random_rotation_point(ansatz.get_parameter_count())
    shots_list = [100 * i for i in range(10, 110, 5)]
    parameter_indices = [i for i in range(ansatz.get_parameter_count())]
    metric = mean_squared_error

    gradient_list = []

    gradient_list.append(
        GradientSampleEstimator(
            FiniteDifferenzen(
                ansatz,
                observable,
                [0.01, -0.01],
                "simulator",
                label="Central Finite Differences h 0.01",
                plot_label="Zentrale Finite Differenzen, h=0.01",
            ),
            5,
        )
    )
    gradient_list.append(
        GradientSampleEstimator(
            SPSAGradient(
                ansatz,
                observable,
                0.01,
                "simulator",
                label="SPSA h 0.01",
                plot_label="SPSA, h=0.01",
            ),
            5,  # ansatz.get_parameter_count(),
        )
    )
    spsa = SPSAGradient(
        ansatz,
        observable,
        0.01,
        "simulator",
        label="SPSA h 0.01",
        plot_label="SPSA, h=0.01",
    )
    spsa.linestyle = "--"
    gradient_list.append(
        GradientSampleEstimator(
            spsa,
            120,  # ansatz.get_parameter_count(),
        )
    )
    compare_gradients_shotslist(
        gradient_list,
        point,
        shots_list,
        parameter_indices,
        metric,
        cm="tab10",
        tab10_indices=[2, 3, 6],
    )


def get_gradlist(ansatz, observable):

    gradient_list = []

    gradient_list.append(
        FiniteDifferenzen(
            ansatz,
            observable,
            [0, 0.01],
            "simulator",
            label="Forward Finite Differences h 0.01",
            plot_label="Vorwärts Finite Differenzen, h=0.01",
        ),
    )

    gradient_list.append(
        FiniteDifferenzen(
            ansatz,
            observable,
            [0, -0.01],
            "simulator",
            label="Backward Finite Differences h 0.01",
            plot_label="Rückwärts Finite Differenzen, h=0.01",
        ),
    )

    gradient_list.append(
        FiniteDifferenzen(
            ansatz,
            observable,
            [0.01, -0.01],
            "simulator",
            label="Central Finite Differences h 0.01",
            plot_label="Zentrale Finite Differenzen, h=0.01",
        ),
    )

    gradient_list.append(
        GradientSampleEstimator(
            SPSAGradient(
                ansatz,
                observable,
                0.01,
                "simulator",
                label="SPSA h 0.01",
                plot_label="SPSA, h=0.01",
            ),
            10,  # ansatz.get_parameter_count(),
        )
    )
    gradient_list.append(
        GeneralParameterShift(
            ansatz,
            observable,
            0.5,
            "simulator",
            label="General Parameter-Shift gamma 0.5",
            plot_label="Allgemeines Parameter-Shift, $\gamma$=0.5",
        ),
    )
    gradient_list.append(
        ParameterShift(
            ansatz,
            observable,
            "simulator",
            label="Simple Parameter-Shift",
            plot_label="Klassisches Parameter-Shift",
        ),
    )
    return gradient_list


def compare_layers_entanglement_plot():
    # Abbildung 6.3
    np.random.seed(40)
    observable = DiagonalObservable(4, [2 * i for i in range(2**4)])

    shots_list = [10 * i for i in range(10, 110, 10)]

    metric = mean_squared_error

    res = np.zeros((2, 2, 6, len(shots_list)))

    ansatz = EfficientSU2Wrapper(qubits=4, layers=1, entanglement="linear")
    point = generate_random_rotation_point(ansatz.get_parameter_count())
    parameter_indices = [i for i in range(ansatz.get_parameter_count())]
    exact = ParameterShift(ansatz, observable, "opflow").evaluate_gradient(
        point, parameter_indices, 0
    )
    gradient_list = get_gradlist(ansatz, observable)

    for i in range(len(gradient_list)):
        gradient_list[i].ansatz = ansatz
        res[0, 0, i, :] = evaluate_metric_shotslist(
            gradient_list[i],
            exact,
            point,
            shots_list,
            parameter_indices,
            mean_squared_error,
        )[0]

    ansatz = EfficientSU2Wrapper(qubits=4, layers=1, entanglement="full")
    point = generate_random_rotation_point(ansatz.get_parameter_count())
    parameter_indices = [i for i in range(ansatz.get_parameter_count())]
    exact = ParameterShift(ansatz, observable, "opflow").evaluate_gradient(
        point, parameter_indices, 0
    )
    gradient_list = get_gradlist(ansatz, observable)
    for i in range(len(gradient_list)):
        gradient_list[i].ansatz = ansatz
        res[1, 0, i, :] = evaluate_metric_shotslist(
            gradient_list[i],
            exact,
            point,
            shots_list,
            parameter_indices,
            mean_squared_error,
        )[0]

    ansatz = EfficientSU2Wrapper(qubits=4, layers=5, entanglement="linear")
    point = generate_random_rotation_point(ansatz.get_parameter_count())
    parameter_indices = [i for i in range(ansatz.get_parameter_count())]
    exact = ParameterShift(ansatz, observable, "opflow").evaluate_gradient(
        point, parameter_indices, 0
    )
    gradient_list = get_gradlist(ansatz, observable)
    for i in range(len(gradient_list)):
        gradient_list[i].ansatz = ansatz
        res[0, 1, i, :] = evaluate_metric_shotslist(
            gradient_list[i],
            exact,
            point,
            shots_list,
            parameter_indices,
            mean_squared_error,
        )[0]

    ansatz = EfficientSU2Wrapper(qubits=4, layers=5, entanglement="full")
    point = generate_random_rotation_point(ansatz.get_parameter_count())
    parameter_indices = [i for i in range(ansatz.get_parameter_count())]
    exact = ParameterShift(ansatz, observable, "opflow").evaluate_gradient(
        point, parameter_indices, 0
    )
    gradient_list = get_gradlist(ansatz, observable)

    for i in range(len(gradient_list)):
        gradient_list[i].ansatz = ansatz
        res[1, 1, i, :] = evaluate_metric_shotslist(
            gradient_list[i],
            exact,
            point,
            shots_list,
            parameter_indices,
            mean_squared_error,
        )[0]

    cols = ["Vollständige Verschränkungschicht", "Lineare Verschränkungschicht"]
    rows = ["l=1", "l=5"]
    n = len(gradient_list)

    colors = [plt.get_cmap("tab10")(1.0 * i / 10) for i in range(10)]
    new_colors = []
    for i in range(len(gradient_list)):
        new_colors.append(colors[i])

    ls = ["-", "--", ":", "-.", "-", "--", ":"]
    mk = ["o", "v", "^", "*", "s", "p", "8"]
    plt.rc(
        "axes",
        prop_cycle=(
            cycler("color", new_colors)
            + cycler("linestyle", ls[:n])
            + cycler("marker", mk[:n])
        ),
    )
    """
    for i in range(6):
        for j in range(2):
            for k in range(2):
                res[k, j, i, :] = np.array(shots_list) / (i + 1)
    """
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)

    # plt.rc("legend", fontsize=15)

    metric_label = get_metric_name(metric, abbreviation=True)

    # ax.ylabel(metric_label)
    # ax.xlabel("Shots")

    for i in range(len(gradient_list)):
        ax[0, 0].plot(
            shots_list,
            res[0, 0, i],
            label=gradient_list[i].plot_label,
        )
        ax[0, 0].set_title("Linear, l=1")  # , fontsize=16)

    for i in range(len(gradient_list)):
        ax[0, 1].plot(
            shots_list,
            res[0, 1, i],
            label=gradient_list[i].plot_label,
        )
        ax[0, 1].set_title("Vollständig, l=1")  # , fontsize=16)

    for i in range(len(gradient_list)):
        ax[1, 0].plot(
            shots_list,
            res[1, 0, i],
            label=gradient_list[i].plot_label,
        )
        ax[1, 0].set_title("Linear, l=5")  # , fontsize=16)

    for i in range(len(gradient_list)):
        ax[1, 1].plot(
            shots_list,
            res[1, 1, i],
            label=gradient_list[i].plot_label,
        )
        ax[1, 1].set_title("Vollständig, l=5")  # , fontsize=16)

    for a in ax[:, 0]:
        a.set_ylabel(metric_label)  # , fontsize=16)
        # a.tick_params(axis="y", labelsize=14)
        # a.tick_params(axis="x", labelsize=14)
    for a in ax[1, :]:
        a.set_xlabel("Shots")  # , fontsize=16)
        # a.tick_params(axis="y", labelsize=14)
        # a.tick_params(axis="x", labelsize=14)
    for a in ax.flat:
        a.grid()

    ax[0, 1].legend(bbox_to_anchor=(1, 1), loc="upper left")

    # plt.title(f"Gradient {metric_label}")
    plt.yscale("log")
    plot_path = "../plots/entangle_l_comp.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    print("Generating Plots..")
    compare_layers_entanglement_plot()  # 6.3
    print("Figure 6.3 finished")
    fd_spsa_plot()  # 6.4
    print("Figure 6.4 finished")
    fd_diff_h_plot()  # 6.5
    print("Figure 6.5 finished")
    fd_konv_plot()  # 6.6
    print("Figure 6.6 finished")
    gPS_PS_plot()  # 6.7
    print("Figure 6.7 finished")
    bias_plot()  # 6.8
    print("Figure 6.8 finished")
    fd_bias_h_plot()  # 6.9
    print("Figure 6.9 finished")
    var_vergleich_plot()  # 6.10
    print("Figure 6.10 finished")
    gPS_gamma_plot()  # 6.11
    print("Figure 6.11 finished")
    var_fd_plot()  # 6.12
    print("Figure 6.12 finished")
