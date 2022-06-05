import numpy as np
import pandas as pd

from datetime import datetime
import mlflow
from .metrics import get_metric_name, mean_squared_error, mean_absolute_error
from collections.abc import Callable
from .plots import (
    plot_comparison_shotslist,
    plot_variance_comparison_shotslist,
    plot_bias_comparison_h_list,
    plot_bias_bins,
)
from .gradient import Gradient, ParameterShift

"""
interfaces to run different experiments. Uses mlflow to track parameter values, hyperparameter information and plots
"""


def run_experiment(
    gradient, parameter, point_list, shots_list, folder_path="../Data/", save=True
):
    def save_experiment():
        parameter_value_list = []
        for point in point_list:
            parameter_value_list.append(point[parameter])
        df = pd.DataFrame(result, index=parameter_value_list, columns=shots_list)
        filename = folder_path + datetime.now().strftime("%d_%m_%Y_%H:%M:%S") + ".csv"
        with open(filename, "w") as fout:
            for key, value in gradient.get_metadata().items():
                fout.write("#" + key + ": " + value + "\n")
            point_string = ""
            for i in range(len(point_list[0])):
                if i == parameter:
                    point_string += "Parameter_Dimension"
                else:
                    point_string += str(point_list[0][i])
                if i < len(point_list[0]) - 1:
                    point_string += ", "
            fout.write("#Evaluation-Point: " + point_string + "\n")
            df.to_csv(fout)

    result = np.zeros((len(point_list), len(shots_list)))

    for i in range(len(point_list)):
        point = point_list[i]
        for j in range(len(shots_list)):
            shots = shots_list[j]

            result[i, j] = gradient.evaluate_gradient(point, [parameter], shots)[
                parameter
            ]

    if save:
        save_experiment()

    return result


def read_experiment_data(filepath, header_rows):
    df = pd.read_csv(filepath, skiprows=header_rows)
    return df.to_numpy()


def evaluate_metric_shotslist(
    gradient: Gradient,
    exact_result: np.ndarray,
    point: np.ndarray,
    shots_list,
    parameter_indices,
    metric: Callable,
):

    gradients = np.zeros((len(shots_list),))

    mlflow.set_tracking_uri("mlruns")

    experiment = mlflow.get_experiment_by_name(
        "Gradient-error for different shotcounts"
    )

    if experiment is None:
        exp_id = mlflow.create_experiment("Gradient-error for different shotcounts")
    else:
        exp_id = experiment.experiment_id
    run_name = (
        exp_id
        + " "
        + gradient.label
        + " "
        + datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    )
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):

        for key, value in gradient.get_metadata().items():
            mlflow.log_param(key, value)

        mlflow.log_param("param_point", point)
        mlflow.log_param("param_list", parameter_indices)

        for i in range(len(shots_list)):
            shots = shots_list[i]
            approx_result = gradient.evaluate_gradient(point, parameter_indices, shots)
            result_i = metric(approx_result, exact_result)
            mlflow.log_metric(get_metric_name(metric), result_i, shots_list[i])
            gradients[i] = result_i

    return gradients, run_name


def compare_gradients_shotslist(
    gradient_list,
    point,
    shots_list,
    parameter_indices,
    metric,
    yscale="log",
    cm="viridis",
    tab10_indices=None,
):

    ansatz = gradient_list[0].ansatz
    observable = gradient_list[0].observable

    exact_result = ParameterShift(ansatz, observable, "opflow").evaluate_gradient(
        point, parameter_indices, 0
    )
    # print(np.average(np.abs(exact_result)))
    result_list = []
    id_list = []
    for gradient in gradient_list:
        result, experiment_id = evaluate_metric_shotslist(
            gradient, exact_result, point, shots_list, parameter_indices, metric
        )
        result_list.append(result)
        id_list.append(experiment_id)

    mlflow.set_tracking_uri("mlruns")

    experiment = mlflow.get_experiment_by_name(
        "Comparison of gradient_errors with different shotcounts"
    )

    if experiment is None:
        exp_id = mlflow.create_experiment(
            "Comparison of gradient_errors with different shotcounts"
        )
    else:
        exp_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=exp_id):

        for key, value in ansatz.get_metadata().items():
            mlflow.log_param(key, value)

        for key, value in observable.get_metadata().items():
            mlflow.log_param(key, value)

        for i in range(len(id_list)):

            mlflow.log_param(gradient_list[i].label + " experiment_id", id_list[i])

        plot_path = plot_comparison_shotslist(
            result_list,
            gradient_list,
            shots_list,
            metric,
            save=True,
            yscale=yscale,
            cm=cm,
            tab10_indices=tab10_indices,
        )
        mlflow.log_artifact(plot_path)


def evaluate_variance_bias_shotslist(
    gradient: Gradient,
    exact_gradients: np.ndarray,
    point: np.ndarray,
    shots_list,
    parameter_indices,
):

    assert gradient.backend == "opflow"

    mlflow.set_tracking_uri("mlruns")

    experiment = mlflow.get_experiment_by_name(
        "Gradient variance and bias for different shotcounts"
    )

    if experiment is None:
        exp_id = mlflow.create_experiment(
            "Gradient variance and bias for different shotcounts"
        )
    else:
        exp_id = experiment.experiment_id

    run_name = (
        exp_id
        + " "
        + gradient.label
        + " "
        + datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    )

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):

        for key, value in gradient.get_metadata().items():
            mlflow.log_param(key, value)

        mlflow.log_param("param_point", point)
        mlflow.log_param("param_list", parameter_indices)

        gradients = gradient.evaluate_gradient(point, parameter_indices, 0)

        bias = gradients - exact_gradients

        variances = gradient.evaluate_gradient_variance(
            point, parameter_indices, shots_list
        )

        mlflow.log_param("bias", bias)

        for i in range(len(shots_list)):
            shots = shots_list[i]
            for j in range(gradient.ansatz.get_parameter_count()):
                mlflow.log_metric(f"variance parameter {j}", variances[i, j], shots)

    return variances, bias, run_name


def compare_variance_bias_shotslist(
    gradient_list,
    point,
    shots_list,
    parameter_indices,
    std_dev=False,
    cm="viridis",
    tab10_indices=None,
):

    ansatz = gradient_list[0].ansatz
    observable = gradient_list[0].observable

    bias_list = []
    variance_list = []
    id_list = []

    exact_gradient = ParameterShift(ansatz, observable, "opflow").evaluate_gradient(
        point, parameter_indices, 0
    )

    for gradient in gradient_list:
        variance, bias, experiment_id = evaluate_variance_bias_shotslist(
            gradient,
            exact_gradient,
            point,
            shots_list,
            parameter_indices,
        )

        if std_dev:
            variance_list.append(np.average(np.sqrt(variance), axis=1))
            bias_list.append(np.average(bias, axis=0))
        else:
            variance_list.append(np.average(variance, axis=1))
            bias_list.append(np.average(np.power(bias, 2), axis=0))

        id_list.append(experiment_id)

    mlflow.set_tracking_uri("mlruns")

    experiment = mlflow.get_experiment_by_name(
        "Comparison of gradient bias and variance with different shotcounts"
    )

    if experiment is None:
        exp_id = mlflow.create_experiment(
            "Comparison of gradient bias and variance with different shotcounts"
        )
    else:
        exp_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=exp_id):

        for key, value in ansatz.get_metadata().items():
            mlflow.log_param(key, value)

        for key, value in observable.get_metadata().items():
            mlflow.log_param(key, value)

        for i in range(len(id_list)):

            mlflow.log_param(gradient_list[i].label + " experiment_id", id_list[i])

        plot_path = plot_variance_comparison_shotslist(
            variance_list,
            bias_list,
            gradient_list,
            shots_list,
            std_dev=std_dev,
            save=True,
            cm=cm,
            tab10_indices=tab10_indices,
        )
        mlflow.log_artifact(plot_path)


def evaluate_sample_variance_shotslist(
    gradient: Gradient, point: np.ndarray, shots_list, parameter_indices
):

    variances = np.zeros((len(shots_list), gradient.ansatz.get_parameter_count()))
    mlflow.set_tracking_uri("mlruns")

    experiment = mlflow.get_experiment_by_name(
        "Gradient sample variance for different shotcounts"
    )

    if experiment is None:
        exp_id = mlflow.create_experiment(
            "Gradient sample variance for different shotcounts"
        )
    else:
        exp_id = experiment.experiment_id

    run_name = (
        exp_id
        + " "
        + gradient.label
        + " "
        + datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    )

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):

        for key, value in gradient.get_metadata().items():
            mlflow.log_param(key, value)

        mlflow.log_param("param_point", point)
        mlflow.log_param("param_list", parameter_indices)

        for i in range(len(shots_list)):
            shots = shots_list[i]
            _, variance_i = gradient.evaluate_gradient_and_sample_variance(
                point, parameter_indices, shots
            )
            variance_i = variance_i / shots
            variances[i, :] = variance_i
            for j in range(variance_i.shape[0]):
                mlflow.log_metric(
                    f"sample variance parameter {j}", variance_i[j], shots
                )

    return variances, run_name


def compare_sample_true_variance_shotslist(
    gradient,
    exact_gradient,
    point,
    shots_list,
    parameter_indices,
    std_dev=False,
    cm="viridis",
    tab10_indices=None,
):

    ansatz = gradient.ansatz
    observable = gradient.observable

    exact_gradient_values = ParameterShift(
        ansatz, observable, "opflow"
    ).evaluate_gradient(point, parameter_indices, 0)

    sample_variances, run_id = evaluate_sample_variance_shotslist(
        gradient, point, shots_list, parameter_indices
    )
    (exact_variances, bias, run_id_exact,) = evaluate_variance_bias_shotslist(
        exact_gradient, exact_gradient_values, point, shots_list, parameter_indices
    )

    if std_dev:
        avg_exact_variances = np.average(np.sqrt(exact_variances), axis=1)
        avg_sample_variances = np.average(np.sqrt(sample_variances), axis=1)
        bias = np.average(bias, axis=0)
    else:
        avg_exact_variances = np.average(exact_variances, axis=1)
        avg_sample_variances = np.average(sample_variances, axis=1)
        bias = np.average(np.power(bias, 2), axis=0)

    mlflow.set_tracking_uri("mlruns")

    experiment = mlflow.get_experiment_by_name(
        "Comparison of gradient exact and sample variance with different shotcounts"
    )

    if experiment is None:
        exp_id = mlflow.create_experiment(
            "Comparison of gradient exact and sample variance with different shotcounts"
        )
    else:
        exp_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=exp_id):

        for key, value in ansatz.get_metadata().items():
            mlflow.log_param(key, value)

        for key, value in observable.get_metadata().items():
            mlflow.log_param(key, value)

        mlflow.log_param(exact_gradient.label + " experiment_id", run_id_exact)
        mlflow.log_param(gradient.label + " experiment_id", run_id)
        print(avg_exact_variances)
        print(avg_sample_variances)
        plot_path = plot_variance_comparison_shotslist(
            [avg_sample_variances, avg_exact_variances],
            [bias, bias],
            [gradient, exact_gradient],
            shots_list,
            std_dev=std_dev,
            save=True,
            cm=cm,
            tab10_indices=tab10_indices,
        )
        mlflow.log_artifact(plot_path)


def evaluate_fd_bias_h_slist(
    gradient_list,
    point: np.ndarray,
    h_list,
    parameter_indices,
    metric,
):

    for gradient in gradient_list:
        assert gradient.backend == "opflow"
    ansatz = gradient_list[0].ansatz
    observable = gradient_list[0].observable

    exact_gradients = ParameterShift(ansatz, observable, "opflow").evaluate_gradient(
        point, parameter_indices, 0
    )

    mlflow.set_tracking_uri("mlruns")

    experiment = mlflow.get_experiment_by_name(
        "Gradient Bias for different shift-sizes"
    )

    if experiment is None:
        exp_id = mlflow.create_experiment("Gradient Bias for different shift-sizes")
    else:
        exp_id = experiment.experiment_id

    run_name = (
        exp_id
        + " "
        + gradient_list[0].label
        + " "
        + datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    )

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):

        for key, value in ansatz.get_metadata().items():
            mlflow.log_param(key, value)

        for key, value in observable.get_metadata().items():
            mlflow.log_param(key, value)

        mlflow.log_param("param_point", point)
        mlflow.log_param("param_list", parameter_indices)

        bias_list = np.zeros((len(gradient_list),))

        for i in range(len(gradient_list)):

            grad = gradient_list[i].evaluate_gradient(point, parameter_indices, 0)
            bias = metric(grad, exact_gradients)
            bias_list[i] = bias

            mlflow.log_metric("bias", bias, i)

        print(h_list)
        print(bias_list)
        plot_path = plot_bias_comparison_h_list(
            bias_list, h_list, metric=metric, save=True
        )

        mlflow.log_artifact(plot_path)


def compare_bias(
    gradient_list, point, parameter_indices, metric, cm="tab10", tab10_indices=None
):
    ansatz = gradient_list[0].ansatz
    observable = gradient_list[0].observable

    bias_list = []

    exact_gradient = ParameterShift(ansatz, observable, "opflow").evaluate_gradient(
        point, parameter_indices, 0
    )

    for gradient in gradient_list:

        res = gradient.evaluate_gradient(point, parameter_indices, 0)

        bias_list.append(metric(res, exact_gradient))

    mlflow.set_tracking_uri("mlruns")

    experiment = mlflow.get_experiment_by_name("Comparison of gradient bias")

    if experiment is None:
        exp_id = mlflow.create_experiment("Comparison of gradient bias")
    else:
        exp_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=exp_id):

        for key, value in ansatz.get_metadata().items():
            mlflow.log_param(key, value)

        for key, value in observable.get_metadata().items():
            mlflow.log_param(key, value)

        plot_path = plot_bias_bins(
            bias_list,
            gradient_list,
            metric=metric,
            save=True,
            cm=cm,
            tab10_indices=tab10_indices,
        )

        mlflow.log_artifact(plot_path)
