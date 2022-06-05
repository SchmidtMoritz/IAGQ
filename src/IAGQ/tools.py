from .gradient import Gradient
import numpy as np
from .observable import Observable


class GradientSampleEstimator:
    """
    Estimates gradient by sampling one gradient function multiple times

    evaluate_gradient implements gradient interface
    """

    def __init__(
        self,
        gradient_func: Gradient,
        num_grad_samples: int,
        estimate_variance=False,
        add_label=True,
    ):
        """
        :param gradient_func: gradient function used
        :param num_grad_samples: amount of samples to draw from gradient_func
        :param estimate_variance: flag if SAMPLE variance of the sampled gradients should also be computed
        :param add_label: flag if add num_grad_samples should be added to labels
        """

        self.gradient_func = gradient_func
        self.num_grad_samples = num_grad_samples
        self.estimate_variance = estimate_variance
        self.label = gradient_func.label
        self.plot_label = gradient_func.plot_label
        if num_grad_samples > 1 and add_label:
            self.label += f" {num_grad_samples} times"
            self.plot_label += f", {num_grad_samples} Ziehungen"

        self.ansatz = gradient_func.ansatz
        self.observable = gradient_func.observable
        self.backend = gradient_func.backend

    def evaluate_gradient(
        self, initial_point: np.ndarray, parameter_indices, shots: int
    ):

        grads = np.zeros((initial_point.shape[0], self.num_grad_samples))
        for i in range(self.num_grad_samples):
            grads[:, i] = self.gradient_func.evaluate_gradient(
                initial_point, parameter_indices, shots
            )

        mean_estimate = np.mean(grads, axis=1)

        # computes SAMPLE variance of the sampled gradients
        # e.g. approximates (sigma^2)/n , with n shots per measurement
        if self.estimate_variance:

            assert self.num_grad_samples > 1

            variance_estimate = np.var(
                grads, axis=1, ddof=1
            )  # ddof=1 for sample variance

            return mean_estimate, variance_estimate
        else:
            return mean_estimate

    def evaluate_gradient_variance(self, initial_point, parameter_indices, shots_list):
        """
        does not implement the evaluate_gradient_variance since shots_list is used

        for non spsa gradients: compute one-shot variance -> divide by n to get variance after n shots

        for spsa: variance_of_expectations unaffected by shots
                 -> computing one shot variance = variance_of_expectations + average_measurement_variance
                    then dividing by n shots falsely shrinks variance_of_expectations part of the sum

        so instead generate variances for entire shots_list correctly in this new function
        :return: variance array of size (len(shots_list),k)
        """

        delta_var = self.gradient_func.evaluate_variance_of_expectations(
            initial_point, parameter_indices, shots_list[-1]
        )
        measurement_var = self.gradient_func.evaluate_average_measurement_variance(
            initial_point, parameter_indices, shots_list[-1]
        )
        variances = np.zeros((len(shots_list), self.ansatz.get_parameter_count()))
        for i in range(len(shots_list)):
            shots = shots_list[i]
            measurement_var_i = measurement_var / shots
            variances[i, :] = (delta_var + measurement_var_i) / self.num_grad_samples
        return variances

    def get_metadata(self):
        meta_dict = self.gradient_func.get_metadata()
        meta_dict["Number of Samples"] = str(self.num_grad_samples)
        return meta_dict


def get_cFD_Bias_Factor(h, r):
    """
    can be used to compute bias of central finite differences as a factor of the actual gradient (arXiv:2106.01388)
    """
    return (np.sin(2 * r * h) / 2 * r * h) - 1


def generate_random_rotation_point(param_count):
    """
    generate random parameter point for rotation gates
    """
    random_point = np.random.randint(0, 10000, (param_count,)).astype(dtype=np.float64)
    random_point *= 2 * np.pi / 10000

    return random_point


def one_param_rotation_grid(input_point, bin_count, parameter_index):
    """
    input point theta (rotation gates):

    generate grid on one parameter theta_i, where entire parameter space {0,2pi} is divided into bin_count amount of bins
    """
    point_list = []
    parameter_value_list = []

    step_size = (2 * np.pi) / (bin_count - 1)
    for i in range(bin_count):
        point = input_point.copy()
        parameter_value = i * step_size
        point[parameter_index] = parameter_value
        parameter_value_list.append(parameter_value)
        point_list.append(point)

    return point_list, parameter_value_list


def eq_superposition_variance(obs: Observable) -> float:
    """
    compute variance of equal superposition as a reference value to other variances
    """
    ew, _ = np.linalg.eig(obs.matrix)
    n = len(ew)

    var = np.sum(np.power(ew, 2) / n) - (np.sum(ew / n) ** 2)

    return var
