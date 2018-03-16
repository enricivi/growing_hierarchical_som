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
        self.position = weight_vector_position

        self.child_map = None
        self.input_dataset = np.empty(shape=(0, self.__weight_map[0].shape[2]), dtype=np.float32)

    def activation(self, data):
        return np.linalg.norm(np.subtract(data, self.weight_vector()), axis=0)

    def needs_child_map(self):
        return self.compute_quantization_error() >= (self.__t2 * self.__zero_quantization_error)

    def compute_quantization_error(self):
        distance_from_whole_dataset = self.activation(self.input_dataset)
        quantization_error = distance_from_whole_dataset.sum()

        if self.__growing_metric is "mqe":
            quantization_error /= self.input_dataset.shape[0]

        return quantization_error

    def weight_distance_from_other_unit(self, unit):
        return np.linalg.norm(self.weight_vector() - unit.weight_vector())

    def weight_vector(self):
        return self.__weight_map[0][self.position]
