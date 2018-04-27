from neuron import NeuronBuilder
from GSOM import GSOM
import numpy as np
from queue import Queue
import progressbar
from multiprocessing import Pool


class GHSOM:
    def __init__(self, input_dataset, t1, t2, learning_rate, decay, gaussian_sigma, growing_metric="qe"):
        self.__input_dataset = input_dataset
        self.__input_dimension = input_dataset.shape[1]

        self.__gaussian_sigma = gaussian_sigma
        self.__decay = decay
        self.__learning_rate = learning_rate

        self.__t1 = t1

        self.__neuron_builder = NeuronBuilder(t2, growing_metric)

    def train(self, epochs_number=15, dataset_percentage=0.25, min_dataset_size=1, seed=None, grow_maxiter=100):
        zero_unit = self.__init_zero_unit(seed=seed)

        neuron_queue = Queue()
        neuron_queue.put(zero_unit)

        pool = Pool(processes=None)

        active_dataset = len(zero_unit.input_dataset)

        bar = progressbar.ProgressBar(max_value=active_dataset, widgets=[
            '[', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.Counter(format='%(value)02d/%(max_value)d'), ') ',
        ])
        bar.update(0)
        while not neuron_queue.empty():
            size = min(neuron_queue.qsize(), pool._processes)
            gmaps = dict()
            for _ in range(size):
                neuron = neuron_queue.get()
                gmaps[neuron] = (pool.apply_async(neuron.child_map.train, (
                    epochs_number,
                    self.__gaussian_sigma,
                    self.__learning_rate,
                    self.__decay,
                    dataset_percentage,
                    min_dataset_size,
                    seed,
                    grow_maxiter
                )))
                active_dataset -= len(neuron.input_dataset)

            for neuron in gmaps:
                gmap = gmaps[neuron].get()
                neuron.child_map = gmap
                neurons_to_expand = filter(lambda _neuron: _neuron.needs_child_map(), gmap.neurons.values())
                for _neuron in neurons_to_expand:
                    _neuron.child_map = self.__build_new_GSOM(
                        _neuron.compute_quantization_error(),
                        _neuron.input_dataset,
                        self.__new_map_weights(_neuron.position, gmap.weights_map[0])
                    )

                    neuron_queue.put(_neuron)

                    active_dataset += len(_neuron.input_dataset)

            bar.update(bar.max_value - active_dataset)

        return zero_unit

    def __init_zero_unit(self, seed):
        zero_unit = self.__neuron_builder.zero_neuron(self.__input_dataset)

        zero_unit.child_map = self.__build_new_GSOM(
            self.__neuron_builder.zero_quantization_error,
            zero_unit.input_dataset,
            self.__calc_initial_random_weights(seed=seed)
        )

        return zero_unit

    # noinspection PyPep8Naming
    def __build_new_GSOM(self, parent_quantization_error, parent_dataset, weights_map):
        return GSOM(
            (2, 2),
            parent_quantization_error,
            self.__t1,
            self.__input_dimension,
            weights_map,
            parent_dataset,
            self.__neuron_builder
        )

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

        child_weights = np.zeros(shape=(2, 2, self.__input_dimension))
        stencil = self.__generate_kernel_stencil(parent_position)
        for child_position in np.ndindex(2, 2):
            child_position = np.asarray(child_position)
            mask = self.__filter_out_of_bound_positions(child_position, stencil, weights_map.shape)

            weight = np.mean(self.__elements_from_positions_list(weights_map, mask), axis=0)

            child_weights[child_position] = weight

        return child_weights

    @staticmethod
    def __elements_from_positions_list(matrix, positions_list):
        return matrix[positions_list[:, 0], positions_list[:, 1]]

    def __filter_out_of_bound_positions(self, child_position, stencil, map_shape):
        return np.asarray(list(filter(lambda pos: self.__check_position(pos, map_shape), stencil + child_position)))

    def __calc_initial_random_weights(self, seed):
        random_generator = np.random.RandomState(seed)
        random_weights = np.zeros(shape=(2, 2, self.__input_dimension))
        for position in np.ndindex(2, 2):
            random_data_item = self.__input_dataset[random_generator.randint(len(self.__input_dataset))]
            random_weights[position] = random_data_item

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
