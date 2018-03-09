from neuron import Neuron
from GSOM import GSOM
import numpy as np


class GHSOM:
    def __init__(self, input_dataset, t1, t2, learning_rate, decay, gaussian_sigma, epoch_number=5,
                 growing_metric="qe"):
        """
        :type epoch_number: The lambda parameter; controls the number of iteration between growing checks
        """
        self.__growing_metric = growing_metric
        self.__gaussian_sigma = gaussian_sigma
        self.__decay = decay
        self.__learning_rate = learning_rate
        self.__t2 = t2
        self.__t1 = t1
        self.__input_dataset = input_dataset
        self.__epoch_number = epoch_number

    def __call__(self, *args, **kwargs):
        zero_unit = Neuron(
            self.__calc_input_mean(),
            None,
            None,
            self.__growing_metric
        )

        zero_unit.input_dataset = self.__input_dataset
        self.__zero_quantization_error = zero_unit.compute_quantization_error()

        zero_unit.child_map = GSOM(
            (2, 2),
            self.__zero_quantization_error,
            self.__t1,
            self.__t2,
            self.__growing_metric,
            self.__calc_initial_random_weights()
        )

        zero_unit.child_map.train(
            self.__input_dataset,
            self.__epoch_number,
            self.__learning_rate,
            self.__decay,
            self.__gaussian_sigma
        )

    def __calc_input_mean(self):
        return self.__input_dataset.mean(axis=0)

    def __calc_initial_random_weights(self):
        weights = dict()
        for idx in range(4):
            position = np.unravel_index(idx, dims=(2, 2))
            random_data_item = self.__input_dataset[np.random.randint(len(self.__input_dataset))]
            weights[position] = random_data_item / np.linalg.norm(random_data_item)
        return weights
