import numpy as np


class Neuron:
    def __init__(self, weight_map, weight_vector_position, zero_quantization_error, t2, growing_metric):
        """
        :type t2: The tau_2 parameter
        :type zero_quantization_error: The quantization error of the layer 0
        """
        self.__growing_metric = growing_metric
        self.__t2 = t2
        self.__zero_quantization_error = zero_quantization_error

        self.__weight_map = weight_map
        self.__weight_idx = weight_vector_position

        self.child_map = None
        self.input_dataset = None

    def activation(self, data):
        return np.linalg.norm(np.subtract(data, self.weight_vector()), axis=0)

    def needs_child_map(self):
        return self.compute_quantization_error() >= (self.__t2 * self.__zero_quantization_error)

    def compute_quantization_error(self):
        assert len(self.input_dataset) != 0, "The unit has not been provided with an input dataset"

        input_dataset = np.asarray(self.input_dataset)
        distance_from_whole_dataset = self.activation(input_dataset)
        quantization_error = distance_from_whole_dataset.sum()

        if self.__growing_metric is "mqe":
            quantization_error /= len(input_dataset)

        return quantization_error

    def weight_distance_from_other_unit(self, unit):
        return np.linalg.norm(self.weight_vector() - unit.weight_vector)

    def weight_vector(self):
        return self.__weight_map[self.__weight_idx]
