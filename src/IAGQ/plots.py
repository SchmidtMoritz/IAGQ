import matplotlib.pyplot as plt
from .metrics import get_metric_name

from datetime import datetime
import numpy as np
from cycler import cycler

"""
plots are put in a plots folder e.g. "./plots/filename.png"
"""


def plot_comparison_shotslist(
    results,
    gradient_list,
    shots_list,
    metric,
    save=False,
    yscale="log",
    cm="viridis",
    tab10_indices=None,
):
    ls = ["-", "--", ":", "-.", "-", "--", ":"]
    mk = ["o", "v", "^", "*", "s", "p", "8"]
    n = len(gradient_list)
    if cm == "tab10":
        colors = [plt.get_cmap(cm)(1.0 * i / 10) for i in range(10)]
        new_colors = []
        for i in tab10_indices:
            new_colors.append(colors[i])

    else:
        new_colors = [plt.get_cmap(cm)(1.0 * i / n) for i in range(n)]

    plt.rc(
        "axes",
        prop_cycle=(
            cycler("color", new_colors)
            + cycler("linestyle", ls[:n])
            + cycler("marker", mk[:n])
        ),
    )

    fig, ax = plt.subplots()

    for i in range(len(results)):
        ax.plot(shots_list, results[i], label=gradient_list[i].plot_label)

    metric_label = get_metric_name(metric, abbreviation=False)
    if cm:
        plt.rcParams["image.cmap"] = cm
    plt.rc("legend", fontsize=15)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", labelsize=14)

    ax.set_xlabel("Shots", fontsize=16)
    ax.set_ylabel(metric_label, fontsize=16)
    plt.yscale(yscale)
    plt.grid()
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    # plt.title(f"Gradient {metric_label}")
    # plt.yscale(yscale)
    if save:
        plot_path = (
            "./plots/Comparison_Shotslist_"
            + datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
            + ".png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        return plot_path
    else:
        plt.show()


def plot_variance_comparison_shotslist(
    variance_list,
    bias_list,
    gradient_list,
    shots_list,
    std_dev=False,
    save=False,
    cm="viridis",
    tab10_indices=None,
):
    n = len(gradient_list)
    if cm == "tab10":
        colors = [plt.get_cmap(cm)(1.0 * i / 10) for i in range(10)]
        new_colors = []
        for i in tab10_indices:
            new_colors.append(colors[i])

    else:
        new_colors = [plt.get_cmap(cm)(1.0 * i / n) for i in range(n)]

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

    fig, ax = plt.subplots()

    if std_dev:
        metric_label = "Bias"
        deviation_label = "Standard Deviation"
    else:
        metric_label = "Bias Squared"
        deviation_label = "Mittlere Quadratische Abweichung"

    assert len(bias_list) == len(variance_list) == len(gradient_list)

    for i in range(len(gradient_list)):

        ls = "--"
        if i % 2:
            ls = ":"
        ax.plot(
            shots_list,
            bias_list[i] + variance_list[i],
            label=gradient_list[i].plot_label,
        )

    plt.rc("legend", fontsize=15)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", labelsize=14)

    ax.set_xlabel("Shots", fontsize=16)
    ax.set_ylabel(f"Erwartete {deviation_label}", fontsize=14)

    plt.grid()
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.yscale("log")
    # plt.title(f"Gradient {metric_label} & {deviation_label}")

    if save:
        plot_path = (
            "./plots/Comparison_Variance_Shotslist_"
            + datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
            + ".png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        return plot_path
    else:
        plt.show()


def plot_bias_bins(
    results, gradient_list, metric, save=False, cm="tab10", tab10_indices=None
):

    ls = ["-", "--", ":", "-.", "-", "--", ":"]
    mk = ["o", "v", "^", "*", "s", "p", "8"]
    n = len(gradient_list)
    if cm == "tab10":
        colors = [plt.get_cmap(cm)(1.0 * i / 10) for i in range(10)]
        new_colors = []
        for i in tab10_indices:
            new_colors.append(colors[i])

    else:
        new_colors = [plt.get_cmap(cm)(1.0 * i / n) for i in range(n)]

    plt.rc(
        "axes",
        prop_cycle=(
            cycler("color", new_colors)
            + cycler("linestyle", ls[:n])
            + cycler("marker", mk[:n])
        ),
    )
    fig, ax = plt.subplots()
    plt.grid()

    ax.set_axisbelow(True)
    ind = np.arange(len(gradient_list))
    for i in range(len(gradient_list)):
        res = np.zeros((len(gradient_list),))
        res[i] = results[i]
        ax.bar(ind, res, label=gradient_list[i].plot_label)

    """
    ax.set_xticks(
        ind,
        labels=[gradient.plot_label for gradient in gradient_list],
        rotation="vertical",
    )
    """
    metric_label = get_metric_name(metric, abbreviation=False)

    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )
    plt.rc("legend", fontsize=15)
    ax.tick_params(axis="y", labelsize=14)

    ax.set_xlabel("Gradientenverfahren", fontsize=16)
    ax.set_ylabel(metric_label, fontsize=16)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")

    # plt.title(f"Gradient Bias")
    plt.yscale("log")
    if save:
        plot_path = (
            "./plots/Comparison_Bias_"
            + datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
            + ".png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        return plot_path
    else:
        plt.show()


def plot_bias_comparison_h_list(results, h_list, metric, save=False):

    fig, ax = plt.subplots()
    ax.invert_xaxis()
    ax.plot(
        h_list,
        results,
        marker="o",
        linestyle="--",
        markersize="7",
        color="blue",
    )

    metric_label = get_metric_name(metric, abbreviation=False)
    plt.xlabel("Abstand h")
    plt.ylabel(metric_label)
    plt.grid()
    plt.xscale("log")
    plt.yscale("log")

    if save:
        plot_path = (
            "./plots/Comparison_Bias_hlist_"
            + datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
            + ".png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        return plot_path
    else:
        plt.show()
