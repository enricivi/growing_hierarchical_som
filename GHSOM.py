from neuron import Neuron
from GSOM import GSOM
import numpy as np
from queue import Queue


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
        zero_unit_map = self.__init_zero_unit()

        map_queue = Queue()
        map_queue.put(zero_unit_map)

        while not map_queue.empty():
            gmap = map_queue.get()
            gmap.train(
                self.__input_dataset,
                self.__epoch_number,
                self.__gaussian_sigma,
                self.__learning_rate,
                self.__decay
            )

            neurons_to_expand = filter(lambda _neuron: _neuron.needs_child_map(), gmap.neurons.values())
            for neuron in neurons_to_expand:
                neuron.child_map = GSOM(
                    (2, 2),
                    neuron.compute_quantization_error(),
                    self.__t1,
                    self.__t2,
                    self.__growing_metric,
                    self.__input_dataset.shape[1],
                    self.__new_map_weights(neuron.position, gmap.weights_map)
                )

                map_queue.put(neuron.child_map)

    def __init_zero_unit(self):
        zero_unit = Neuron(
            np.reshape(self.__calc_input_mean(), newshape=(1, 1, self.__input_dataset.shape[1])),
            (0, 0),
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
            self.__input_dataset.shape[1],
            self.__calc_initial_random_weights()
        )

        return zero_unit.child_map

    def __calc_input_mean(self):
        return self.__input_dataset.mean(axis=0)

    def __calc_initial_random_weights(self):
        weights = dict()
        for idx in range(4):
            position = np.unravel_index(idx, dims=(2, 2))
            random_data_item = self.__input_dataset[np.random.randint(len(self.__input_dataset))]
            weights[position] = random_data_item / np.linalg.norm(random_data_item)
        return weights

    def __new_map_weights(self, parent_position, weights_map):
        """
         ______ ______ ______
        |      |      |      |         child (2x2)
        | pnfp |      |      |          ______ ______
        |______|______|______|         |      |      |
        |      |      |      |         |(0,0) |(0,1) |
        |      |parent|      |  ---->  |______|______|
        |______|______|______|         |      |      |
        |      |      |      |         |(1,0) |(1,1) |
        |      |      |      |         |______|______|
        |______|______|______|
        """

        child_weights = np.zeros(shape=(2, 2, self.__input_dataset.shape[1]))
        stencil = self.__generate_kernel_stencil(parent_position)
        for child_position in np.ndindex(2, 2):
            child_position = np.asarray(child_position)
            mask = filter(self.__check_position, stencil + child_position)
            weight = np.mean(weights_map[mask[:, 0], mask[:, 1]], axis=0)
            weight /= np.linalg.norm(weight)

            child_weights[child_position] = weight

        return child_weights

    def __generate_kernel_stencil(self, parent_position):
        row, col = parent_position
        return np.asarray([
            (r, c)
            for r in range(row - 1, row + 1)
            for c in range(col - 1, col + 1)
        ])

    def __check_position(self, position, map_size):
        row, col = position
        map_rows, map_cols = map_size
        return (row >= 0 and col >=0) and (row < map_rows and col < map_cols)
