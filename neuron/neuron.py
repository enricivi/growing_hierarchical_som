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

        self.previous_count = 0
        self.current_count = 0

        self.child_map = None
        self.input_dataset = self.__init_empty_dataset()

    def activation(self, data):
        activation = 0
        if len(data.shape) == 1:  # if data is a vector
            activation = np.linalg.norm(np.subtract(data, self.weight_vector()), ord=2)
        else:
            activation = np.linalg.norm(np.subtract(data, self.weight_vector()), ord=2, axis=1)
        return activation

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

    def has_dataset(self):
        return len(self.input_dataset) != 0

    def replace_dataset(self, data_item):
        self.current_count = len(data_item)
        self.input_dataset = data_item

    def clear_dataset(self):
        self.previous_count = self.current_count
        self.current_count = 0
        self.input_dataset = self.__init_empty_dataset()

    def has_changed_from_previous_epoch(self):
        return self.previous_count != self.current_count

    def __init_empty_dataset(self):
        input_dimension = self.__weight_map[0].shape[2]
        return np.empty(shape=(0, input_dimension), dtype=np.float32)

    def __repr__(self, level=0):
        gap = '\t'*level
        _printable = "{}position {} -- map dimensions {} -- input dataset {} element(s) -- level {} \n".format(
            gap,
            self.position,
            self.__weight_map[0].shape,
            self.input_dataset.shape[0],
            level
        )

        if self.child_map is not None:
            for neuron in self.child_map.neurons.values():
                _printable += neuron.__repr__(level+1)
        return _printable
