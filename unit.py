import numpy as np


class Unit:
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
        self.input_dataset = None

    def needs_child_map(self):
        # global stopping criterion
        raise NotImplementedError
        return False

    def calc_quantization_error(self):
        assert self.input_dataset is not None, "The unit has not been provided with an input dataset"
        distance_from_whole_dataset = np.linalg.norm(self.input_dataset - self.__weight_vector, axis=0)
        quantization_error = distance_from_whole_dataset.sum()
        if self.__growing_metric is "mqe":
            quantization_error /= len(self.input_dataset)

        return quantization_error

    def distance_from_other_unit(self, unit):
        # 2-norm
        raise NotImplementedError

    def __distance_from_input_data_item(self, data_item):
        return np.linalg.norm((self.__weight_vector - data_item))
