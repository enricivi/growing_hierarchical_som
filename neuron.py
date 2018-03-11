from typing import List

import numpy as np


class Neuron:
    input_dataset = list()  # type: List

    def __init__(self, weight_vector, zero_quantization_error, t2, growing_metric):
        """
        :type t2: The tau_2 parameter
        :type zero_quantization_error: The quantization error of the layer 0
        :type weight_vector: the initial weight vector for the unit - m_i
        """
        self.__growing_metric = growing_metric
        self.__t2 = t2
        self.__zero_quantization_error = zero_quantization_error
        self.__weight_vector = weight_vector

        self.child_map = None

    def activation(self, data):
        return np.linalg.norm(data - self.__weight_vector)

    def needs_child_map(self):
        return self.compute_quantization_error() >= (self.__t2 * self.__zero_quantization_error)

    def compute_quantization_error(self):
        assert len(self.input_dataset) != 0, "The unit has not been provided with an input dataset"

        input_dataset = np.asarray(self.input_dataset)
        distance_from_whole_dataset = np.linalg.norm(self.activation(input_dataset), axis=1)
        quantization_error = distance_from_whole_dataset.sum()

        if self.__growing_metric is "mqe":
            quantization_error /= len(input_dataset)

        return quantization_error

    def weight_distance_from_other_unit(self, unit):
        return np.linalg.norm(self.__weight_vector - unit.__weight_vector)
