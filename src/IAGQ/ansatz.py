import abc

from qiskit import *
from abc import ABC
from qiskit.circuit import ParameterVector
from math import floor
from typing import List, Union, Callable
from qiskit_nature.circuit.library import UCCSD, HartreeFock
from qiskit.circuit.library.n_local import EfficientSU2

# https://docs.python.org/3/library/functions.html#property vlt ne idee


class Ansatz(ABC):
    """abstract ansatz base class"""

    def __init__(self, qubits):
        self.qubits = qubits
        self.qr = QuantumRegister(self.qubits)
        self.circuit = QuantumCircuit(self.qr)
        self.parameters = None
        self.parameter_count = None

    def get_parameters(self):
        return self.parameters

    def get_circuit(self):
        return self.circuit

    @abc.abstractmethod
    def get_bound_circuit(self, parameter_values):
        """returns circuit with fixed parameter values, interface used by circuit runner"""
        return NotImplemented

    @abc.abstractmethod
    def is_parameter_shiftable(self):
        """return true of circuit gates are parameter shiftable: Generator with two distinct eigenvalues +/- r"""
        return NotImplemented

    @abc.abstractmethod
    def get_r_values(self):
        """return list of r values of gates. only if ansatz is parameter shiftable"""
        return NotImplemented

    def get_parameter_count(self):
        return self.parameter_count

    @abc.abstractmethod
    def get_metadata(self):
        """return dictionary with meta data to be displayed in mlflow"""
        meta_dict = {"Qubits": str(self.qubits)}
        return meta_dict

    def draw(self):
        self.circuit.draw(output="mpl").show()


class LayeredHardwareEfficientAnsatz(Ansatz):
    """ansatz based on arXiv:2004.01372"""

    def __init__(self, qubits, layers):
        super().__init__(qubits)

        n_Bs = floor(qubits / 2)
        n_Params = 4
        self.parameter_count = n_Params * layers * n_Bs
        self.parameters = ParameterVector("theta", self.parameter_count)
        self.layers = layers

        # generates alternating layered structure of B_mue blocks: see arXiv:2004.01372
        for l in range(layers):
            for i in range(0, qubits, 2):
                theta_indices = [
                    k
                    for k in range(
                        (l * n_Bs + floor(i / 2)) * n_Params,
                        (l * n_Bs + floor(i / 2)) * n_Params + n_Params,
                    )
                ]
                # print(theta_indices,l,i)
                if l % 2:
                    i = i + 1
                self.b_mue1(self.qr[i], self.qr[(i + 1) % qubits], theta_indices)

    def b_mue1(
        self,
        qubit1: int,
        qubit2: int,
        theta_indices: List[int],
    ):
        """
        generates B_mue block

        :param qubit1: first qubit
        :param qubit2: second qubit
        :param theta_indices: indices of the parameters relevant for the specific block
        :return: expanded circuit with new parametric block
        """
        self.circuit.ry(self.parameters[theta_indices[0]], qubit1)
        self.circuit.ry(self.parameters[theta_indices[1]], qubit2)
        self.circuit.cz(qubit1, qubit2)
        self.circuit.ry(self.parameters[theta_indices[2]], qubit1)
        self.circuit.ry(self.parameters[theta_indices[3]], qubit2)

    def get_bound_circuit(self, parameter_values):
        return self.circuit.bind_parameters({self.parameters: parameter_values})

    def is_parameter_shiftable(self):
        return True

    def get_r_values(self):
        return [0.5 for i in range(self.parameter_count)]

    def get_metadata(self):
        meta_dict = super().get_metadata()
        meta_dict["Ansatz"] = "Layered Hardware Efficient Ansatz (VQSE)"
        meta_dict["Ansatz-Layers"] = str(self.layers)

        return meta_dict


class CR_Euler_Ansatz(Ansatz):
    """
    ansatz similar to VQE ansatz from: arXiv:1304.3061

     observable specific cross resonance gates replaced by fixed RZX gates

     zero layer: RX,RZ

     after that
     every layer: linear CR(replaced by fixed RZX) -> RZ,RX,RZ
    """

    def __init__(self, qubits, layers):
        super().__init__(qubits)

        assert layers >= 1
        self.layers = layers

        params_per_layer = 3 * self.qubits  # three rotations per qubit per layer
        first_offset = 2 * self.qubits  # only two rotations in first layer
        self.parameter_count = self.layers * params_per_layer + first_offset
        self.parameters = ParameterVector("theta", self.parameter_count)
        parameter_indices = [i for i in range(self.parameter_count)]

        self.add_first_layer(parameter_indices[:first_offset])

        for i in range(self.layers):
            self.add_layer(
                parameter_indices[
                    first_offset
                    + params_per_layer * i : first_offset
                    + params_per_layer * (i + 1)
                ]
            )

    def add_first_layer(self, theta_indices):
        assert len(theta_indices) == self.qubits * 2

        first_rotation_indices = theta_indices[: self.qubits * 2]

        for i in range(self.qubits):
            self.add_first_rotation(i, first_rotation_indices[2 * i : 2 * (i + 1)])

    def add_layer(self, theta_indices):
        assert len(theta_indices) == self.qubits * 3

        self.circuit.barrier()
        self.add_entangler()
        self.circuit.barrier()

        for i in range(self.qubits):
            self.add_euler_rotation(i, theta_indices[3 * i : 3 * (i + 1)])

    def add_euler_rotation(self, qubit, theta_indices):
        self.circuit.rz(self.parameters[theta_indices[0]], qubit)
        self.circuit.rx(self.parameters[theta_indices[1]], qubit)
        self.circuit.rz(self.parameters[theta_indices[2]], qubit)

    def add_first_rotation(self, qubit, theta_indices):
        self.circuit.rx(self.parameters[theta_indices[0]], qubit)
        self.circuit.rz(self.parameters[theta_indices[1]], qubit)

    def add_entangler(self):
        for i in range(self.qubits - 1):
            self.circuit.ecr(i, i + 1)

    def get_bound_circuit(self, parameter_values):
        return self.circuit.bind_parameters({self.parameters: parameter_values})

    def is_parameter_shiftable(self):
        return True

    def get_r_values(self):
        return [0.5 for i in range(self.parameter_count)]

    def get_metadata(self):
        meta_dict = super().get_metadata()
        meta_dict["Ansatz"] = "VQE Ansatz CR+EulerRotations (1704.05018)"
        meta_dict["Ansatz-Layers"] = str(self.layers)

        return meta_dict


class EfficientSU2Wrapper(Ansatz):
    """
    wrapper class for Qiskits EfficientSU2 class

    ref: https://qiskit.org/documentation/stubs/qiskit.circuit.library.EfficientSU2.html
    """

    def __init__(self, qubits, entanglement, layers):
        super().__init__(qubits)
        self.layers = layers
        self.eSU2 = EfficientSU2(
            num_qubits=qubits,
            su2_gates=["rx", "rz"],
            entanglement=entanglement,
            reps=self.layers,
        )
        self.circuit.compose(self.eSU2, inplace=True)
        self.parameter_count = self.eSU2.num_parameters
        self.parameters = self.eSU2.ordered_parameters

    def get_bound_circuit(self, parameter_values):
        return self.circuit.copy().assign_parameters(parameter_values)

    def is_parameter_shiftable(self):
        return True

    def get_r_values(self):
        return [0.5 for i in range(self.parameter_count)]

    def get_metadata(self):
        meta_dict = super().get_metadata()
        meta_dict["Ansatz"] = "Qiskit Harware-Efficient SU2"
        meta_dict["Ansatz-Layers"] = str(self.layers)
        meta_dict["SU2-Gates"] = "rx, rz"
        meta_dict["Entanglement"] = self.eSU2.entanglement

        return meta_dict

    def draw(self):
        self.circuit.decompose().draw(output="mpl").show()
