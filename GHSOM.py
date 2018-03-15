from neuron import NeuronBuilder
from GSOM import GSOM
import numpy as np
from queue import Queue


class GHSOM:
    def __init__(self, t1, t2, learning_rate, decay, gaussian_sigma, epoch_number=5,
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
        self.__epoch_number = epoch_number

    def __call__(self, input_dataset, *args, **kwargs):
        zero_unit = self.__init_zero_unit(input_dataset)
        zero_unit_child_map = zero_unit.child_map

        map_queue = Queue()
        map_queue.put(zero_unit_child_map)

        while not map_queue.empty():
            gmap = map_queue.get()
            gmap.train(
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
                    input_dataset.shape[1],
                    self.__new_map_weights(neuron.position, gmap.weights_map, input_dataset.shape[1]),
                    neuron.input_dataset
                )

                map_queue.put(neuron.child_map)

        return zero_unit

    def __init_zero_unit(self, input_dataset):
        zero_unit = NeuronBuilder.zero_neuron(
            np.reshape(self.__calc_input_mean(input_dataset), newshape=(1, 1, input_dataset.shape[1])),
            input_dataset,
            self.__t2,
            self.__growing_metric
        )

        zero_unit.child_map = GSOM(
            (2, 2),
            NeuronBuilder.get_zero_quantization_error(),
            self.__t1,
            self.__t2,
            self.__growing_metric,
            input_dataset.shape[1],
            self.__calc_initial_random_weights(input_dataset),
            zero_unit.input_dataset
        )

        return zero_unit

    def __new_map_weights(self, parent_position, weights_map, features_length):
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

        child_weights = np.zeros(shape=(2, 2, features_length))
        stencil = self.__generate_kernel_stencil(parent_position)
        for child_position in np.ndindex(2, 2):
            mask = np.asarray([s for s in stencil if self.__check_position(s, weights_map.shape)])
            weight = np.mean(weights_map[mask[:, 0], mask[:, 1]], axis=0)
            weight /= np.linalg.norm(weight)

            child_weights[child_position] = weight

        return child_weights

    @staticmethod
    def __calc_input_mean(input_dataset):
        return input_dataset.mean(axis=0)

    @staticmethod
    def __calc_initial_random_weights(input_dataset):
        random_weights = np.zeros(shape=(2, 2, input_dataset.shape[1]))
        for position in np.ndindex(2, 2):
            random_data_item = input_dataset[np.random.randint(len(input_dataset))]
            random_weights[position] = random_data_item / np.linalg.norm(random_data_item)

        return random_weights

    @staticmethod
    def __generate_kernel_stencil(parent_position):
        row, col = parent_position
        return np.asarray([
            (r, c)
            for r in range(row - 1, row + 1)
            for c in range(col - 1, col + 1)
        ])

    @staticmethod
    def __check_position(position, map_shape):
        row, col = position
        map_rows, map_cols = map_shape[0], map_shape[1]
        return (row >= 0 and col >= 0) and (row < map_rows and col < map_cols)
