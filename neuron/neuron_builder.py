from neuron.neuron import Neuron
import numpy as np


class NeuronBuilder:
    zero_quantization_error = None

    def __init__(self, tau_2, growing_metric):
        self.__growing_metric = growing_metric
        self.__tau_2 = tau_2

    def new_neuron(self, weights_map, position):
        assert self.zero_quantization_error is not None, "Zero quantization error has not been set yet"

        return Neuron(
            weights_map,
            position,
            self.zero_quantization_error,
            self.__tau_2,
            self.__growing_metric
        )

    def zero_neuron(self, input_dataset):
        input_dimension = input_dataset.shape[1]
        zero_neuron = Neuron(
            [self.__calc_input_mean(input_dataset).reshape(1, 1, input_dimension)],
            (0, 0),
            None,
            None,
            self.__growing_metric
        )
        zero_neuron.input_dataset = input_dataset

        self.zero_quantization_error = zero_neuron.compute_quantization_error()

        return zero_neuron

    @staticmethod
    def __calc_input_mean(input_dataset):
        return input_dataset.mean(axis=0)
