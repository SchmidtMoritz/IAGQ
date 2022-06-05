import abc
from abc import ABC
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import axes_grid1
from qiskit.opflow import I, X, Z, Y, SummedOp


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


class Observable(ABC):
    """
    abstract observable base class
    """

    @abc.abstractmethod
    def __init__(self, qubits):
        self.matrix = None
        self.qubits = qubits

    def get_matrix(self):
        """
        interface used by circuitrunner to use observable for measurements
        """
        return self.matrix

    def plot(self):
        fig, ax = plt.subplots(1, 2, figsize=(9, 5))

        im = ax[0].imshow(self.matrix.real, cmap="coolwarm")
        im2 = ax[1].imshow(self.matrix.imag, cmap="coolwarm")

        ax[0].set_title("Real Part")
        ax[1].set_title("Imaginary Part")

        add_colorbar(im)
        add_colorbar(im2)

        fig.suptitle("Observable")
        plt.show()

    @abc.abstractmethod
    def is_stochastically_measurable(self):
        """
        if observable is given as linear combination of pauli strings
        circuits can be measured stochastically based on arXiv:1910.01155

        method has to be implemented accordingly
        """
        return NotImplemented

    @abc.abstractmethod
    def get_metadata(self):
        """return dictionary with meta data to be displayed in mlflow"""
        return NotImplemented


class StochasticallyMeasurableObservable(Observable):
    """abstract baseclass for stochastically measurable observables e.g. O is given as a linear combination"""

    @abc.abstractmethod
    def __init__(self, qubits, coefficients, pauli_strings, stochastic=0):

        super().__init__(qubits)
        self.stochastic = stochastic  # amount of terms to be sampled for measurement, 0-> no stochastic measurement
        self.coefficients = (
            coefficients  # list of coefficients of baseelements of lin. comb.
        )
        self.pauli_strings = pauli_strings  # list of occurring paulistrings

    def get_matrix(self):

        if self.stochastic > 0:  # if stochastic measurement

            M = len(self.coefficients)

            # randomly choose terms and coefficients
            random_indices = np.random.choice(
                range(M), size=self.stochastic, replace=False
            )

            H = SummedOp([])

            # construct new observable out of sampled terms
            for i in random_indices:

                H += (
                    (M / self.stochastic) * self.coefficients[i] * self.pauli_strings[i]
                )

            return H.to_matrix()

        else:  # if no stochastic measurement
            return super().get_matrix()

    def is_stochastically_measurable(self):
        return True


class DiagonalObservable(Observable):
    """diagonal matrix observable"""

    def __init__(self, qubits, weights):
        super().__init__(qubits)
        self.weights = weights  # list of diagonal elements

        assert len(weights) == 2**qubits

        self.matrix = np.eye(2**qubits, dtype=complex)
        for i in range(2**qubits):
            self.matrix[i, i] = weights[i]

    def is_stochastically_measurable(self):
        return False

    def get_metadata(self):
        weight_string = ""
        for i in range(len(self.weights)):
            weight = self.weights[i]
            weight_string += str(weight)
            if i < len(self.weights) - 1:
                weight_string += ", "

        meta_dict = {"Observable": "Eye-Matrix", "Observable-Weights": weight_string}
        return meta_dict


class VQE_HeHObservable(StochasticallyMeasurableObservable):
    """
    VQE Observable ref:arXiv:1304.3061
    """

    def __init__(self, r, stochastic=0):
        self.r = r
        r_values = [0.05, 0.1, 1, 2, 3]

        assert self.r in r_values

        coefficients = []
        if self.r == 0.05:
            coefficients = [
                33.9557,
                -0.1515,
                -2.4784,
                -0.1515,
                0.1412,
                0.1515,
                -2.4784,
                0.1515,
                0.2746,
            ]
        elif self.r == 0.1:
            coefficients = [
                13.3605,
                -0.1626,
                -2.4368,
                -0.1626,
                0.2097,
                0.1626,
                -2.4368,
                0.1626,
                0.2081,
            ]
        elif self.r == 1:
            coefficients = [
                -3.9734,
                -0.2385,
                -1.0052,
                -0.2385,
                0.2343,
                0.2385,
                -1.0052,
                0.2385,
                0.2779,
            ]
        elif self.r == 2:
            coefficients = [
                -4.1347,
                -0.1119,
                -1.0605,
                -0.1119,
                0.0212,
                0.1119,
                -1.0605,
                0.1119,
                0.6342,
            ]
        elif self.r == 3:
            coefficients = [
                -4.0578,
                -0.0159,
                -1.1482,
                -0.0159,
                0.0004,
                0.0159,
                -1.1482,
                0.0159,
                0.7385,
            ]

        pauli_strings = [
            (I ^ I),
            (I ^ X),
            (I ^ Z),
            (X ^ I),
            (X ^ X),
            (X ^ Z),
            (Z ^ I),
            (Z ^ X),
            (Z ^ Z),
        ]

        assert len(coefficients) == len(pauli_strings)

        super().__init__(2, coefficients, pauli_strings, stochastic)

        H = SummedOp([])
        for i in range(len(coefficients)):
            H += coefficients[i] * pauli_strings[i]

        self.matrix = H.to_matrix()

    def is_stochastically_measurable(self):
        return True

    def get_metadata(self):
        meta_dict = {"Observable": "VQE HeH+ Observable", "R": self.r}
        return meta_dict


class VQE_H2_Observable(StochasticallyMeasurableObservable):
    """
    VQE Observable ref:arXiv:1704.05018
    """

    def __init__(self, stochastic=0):

        coefficients = [0.011280, 0.397936, 0.397936, 0.180931]

        pauli_strings = [(Z ^ Z), (Z ^ I), (I ^ Z), (X ^ X)]

        assert len(coefficients) == len(pauli_strings)

        super().__init__(2, coefficients, pauli_strings, stochastic)

        H = SummedOp([])

        for i in range(len(coefficients)):
            H += coefficients[i] * pauli_strings[i]

        self.matrix = H.to_matrix()

    def get_metadata(self):
        meta_dict = {"Observable": "VQE H2 Observable"}
        return meta_dict


class VQE_LiH_Observable(StochasticallyMeasurableObservable):
    """
    VQE Observable ref:arXiv:1704.05018
    """

    def __init__(self, stochastic=0):

        coefficients = [
            -0.096022,
            -0.206128,
            0.364746,
            0.096022,
            -0.206128,
            -0.364746,
            -0.145438,
            0.056040,
            0.110811,
            -0.056040,
            0.080334,
            0.063673,
            0.110811,
            -0.063673,
            -0.095216,
            0.008195,
            0.001271,
            -0.012585,
            0.012585,
            0.012585,
            0.012585,
            -0.002667,
            -0.002667,
            0.002667,
            0.002667,
            0.007265,
            -0.007265,
            0.007265,
            0.007265,
            -0.008195,
            0.008195,
            -0.029640,
            0.002792,
            -0.029640,
            0.002792,
            -0.008195,
            -0.001271,
            -0.008195,
            0.028926,
            0.007499,
            -0.001271,
            0.007499,
            0.009327,
            -0.001271,
            0.001271,
            0.008025,
            0.029640,
            0.029640,
            0.028926,
            0.002792,
            -0.002792,
            -0.016781,
            0.016781,
            -0.016781,
            -0.016781,
            -0.009327,
            0.009327,
            -0.009327,
            -0.011962,
            -0.011962,
            0.000247,
            0.000247,
            0.039155,
            -0.002895,
            -0.009769,
            -0.024280,
            -0.008025,
            -0.039155,
            0.002895,
            0.024280,
            -0.011962,
            0.011962,
            -0.000247,
            0.000247,
            -0.039155,
            -0.002895,
            0.024280,
            -0.009769,
            0.008025,
            0.039155,
            0.002895,
            -0.024280,
            -0.008195,
            -0.001271,
            0.008195,
            0.008195,
            -0.028926,
            -0.007499,
            -0.028926,
            -0.007499,
            -0.007499,
            0.007499,
            0.009769,
            -0.001271,
            -0.001271,
            0.008025,
            0.007499,
            -0.007499,
            -0.009769,
        ]

        pauli_strings = [
            (Z ^ I ^ I ^ I),
            (Z ^ Z ^ I ^ I),
            (I ^ Z ^ I ^ I),
            (I ^ I ^ Z ^ I),
            (I ^ I ^ Z ^ Z),
            (I ^ I ^ I ^ Z),
            (Z ^ I ^ Z ^ I),
            Z ^ I ^ Z ^ Z,
            Z ^ I ^ I ^ Z,
            Z ^ Z ^ Z ^ I,
            Z ^ Z ^ Z ^ Z,
            Z ^ Z ^ I ^ Z,
            I ^ Z ^ Z ^ I,
            I ^ Z ^ Z ^ Z,
            I ^ Z ^ I ^ Z,
            X ^ Z ^ X ^ X,
            X ^ Z ^ I ^ X,
            X ^ Z ^ I ^ I,
            X ^ I ^ I ^ I,
            I ^ I ^ X ^ Z,
            I ^ I ^ X ^ I,
            X ^ Z ^ X ^ Z,
            X ^ Z ^ X ^ I,
            X ^ I ^ X ^ Z,
            X ^ I ^ X ^ I,
            X ^ Z ^ I ^ Z,
            X ^ I ^ I ^ Z,
            I ^ Z ^ X ^ Z,
            I ^ Z ^ X ^ I,
            X ^ Z ^ Y ^ Y,
            X ^ I ^ Y ^ Y,
            X ^ X ^ I ^ I,
            I ^ X ^ I ^ I,
            I ^ I ^ X ^ X,
            I ^ I ^ I ^ X,
            X ^ I ^ X ^ X,
            X ^ I ^ I ^ X,
            X ^ X ^ X ^ I,
            X ^ X ^ X ^ X,
            X ^ X ^ I ^ X,
            I ^ X ^ X ^ I,
            I ^ X ^ X ^ X,
            I ^ X ^ I ^ X,
            X ^ Z ^ Z ^ X,
            X ^ I ^ Z ^ X,
            I ^ Z ^ Z ^ X,
            Y ^ Y ^ I ^ I,
            I ^ I ^ Y ^ Y,
            Y ^ Y ^ Y ^ Y,
            Z ^ X ^ I ^ I,
            I ^ I ^ Z ^ X,
            Z ^ I ^ Z ^ X,
            Z ^ I ^ I ^ X,
            Z ^ X ^ Z ^ I,
            I ^ X ^ Z ^ I,
            Z ^ X ^ Z ^ X,
            Z ^ X ^ I ^ X,
            I ^ X ^ Z ^ X,
            Z ^ I ^ X ^ Z,
            Z ^ I ^ X ^ I,
            Z ^ Z ^ X ^ Z,
            Z ^ Z ^ X ^ I,
            Z ^ I ^ X ^ X,
            Z ^ Z ^ X ^ X,
            Z ^ Z ^ I ^ X,
            I ^ Z ^ X ^ X,
            I ^ Z ^ I ^ X,
            Z ^ I ^ Y ^ Y,
            Z ^ Z ^ Y ^ Y,
            I ^ Z ^ Y ^ Y,
            X ^ Z ^ Z ^ I,
            X ^ I ^ Z ^ I,
            X ^ Z ^ Z ^ Z,
            X ^ I ^ Z ^ Z,
            X ^ X ^ Z ^ I,
            X ^ X ^ Z ^ Z,
            X ^ X ^ I ^ Z,
            I ^ X ^ Z ^ Z,
            I ^ X ^ I ^ Z,
            Y ^ Y ^ Z ^ I,
            Y ^ Y ^ Z ^ Z,
            Y ^ Y ^ I ^ Z,
            X ^ X ^ X ^ Z,
            (I ^ X ^ X ^ Z),
            (Y ^ Y ^ X ^ Z),
            (Y ^ Y ^ X ^ I),
            (X ^ X ^ Y ^ Y),
            (I ^ X ^ Y ^ Y),
            Y ^ Y ^ X ^ X,
            (Y ^ Y ^ I ^ X),
            X ^ X ^ Z ^ X,
            Y ^ Y ^ Z ^ X,
            Z ^ Z ^ Z ^ X,
            Z ^ X ^ X ^ Z,
            Z ^ X ^ X ^ I,
            Z ^ X ^ I ^ Z,
            Z ^ X ^ X ^ X,
            Z ^ X ^ Y ^ Y,
            Z ^ X ^ Z ^ Z,
        ]

        assert len(coefficients) == len(pauli_strings)

        super().__init__(4, coefficients, pauli_strings, stochastic)

        H = SummedOp([])

        for i in range(len(coefficients)):
            H += coefficients[i] * pauli_strings[i]

        A = H.to_matrix()

        self.matrix = H.to_matrix()

    def get_metadata(self):
        meta_dict = {"Observable": "VQE LiH Observable"}
        return meta_dict
