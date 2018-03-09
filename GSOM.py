from neuron import Neuron
import numpy as np


class GSOM:
    def __init__(self, initial_map_size, parent_quantization_error, t1, t2, growing_metric, weights_vectors_dict=None):
        self.__t1 = t1
        self.__t2 = t2
        self.__parent_quantization_error = parent_quantization_error
        self.__map_size = initial_map_size
        self.__growing_metric = growing_metric

        # i'm not so sure...
        self.neurons_map = np.asarray(
            a=[[None for _ in range(initial_map_size[0])] for _ in range(initial_map_size[1])],
            dtype=object
        )
        if weights_vectors_dict is not None:
            for position, weight in weights_vectors_dict.items():
                self.neurons_map[position] = Neuron(weight, self.__parent_quantization_error, self.__t2, growing_metric)

    def winner_idx(self, data):
        activations = np.empty(shape=self.neurons_map.shape, dtype=np.float32)

        activations_iter = np.nditer(activations, flags=['multi_index'])
        while not activations_iter.finished:
            activations[activations_iter.multi_index] = self.neurons_map[activations_iter.multi_index].activation(data)
            activations_iter.iternext()

        return np.unravel_index(activations.min(), dims=activations.shape)

    def train(self, input_dataset, epochs, learning_rate, decay, gaussian_sigma):
        lr = learning_rate
        sigma = gaussian_sigma
        can_grow = self.__can_grow(input_dataset)
        iter = 1
        while can_grow:
            # updating weights
            data = input_dataset[np.random.randint(len(input_dataset))]
            gauss_kernel = self.__gaussian_kernel(self.winner_idx(data), sigma)
            self.__update_neurons(data, gauss_kernel, lr)
            # updating lr and sigma
            lr *= decay
            sigma *= decay
            # updating map dimensions
            if (iter % epochs) == 0:
                can_grow = self.__can_grow(input_dataset)
                if can_grow:
                    self.grow()
                lr = learning_rate
                sigma = gaussian_sigma
            iter += 1

    def __update_neurons(self, data, gaussian_kernel, learning_rate):
        # updating neurons weight
        map_iter = np.nditer(self.neurons_map, flags=['multi_index'])
        while not map_iter.finished:
            weight = self.neurons_map[map_iter.multi_index].__weight_vector
            weight += learning_rate * gaussian_kernel[map_iter.multi_index] * (data - weight)
            weight /= np.linalg.norm(weight)
            self.neurons_map[map_iter.multi_index].__weight_vector = weight
            map_iter.iternext()

    def __gaussian_kernel(self, winner_neuron, gaussian_sigma):
        # TODO: kernel area != 1 (multiply kernel and A = 1/[2*pi*(gaussian_sigma**2)] to obtain a unit area). probably it's not necessary
        # computing gaussian kernel
        s = 2 * (gaussian_sigma) ** 2
        gauss_y = np.power(np.asarray(range(self.neurons_map.shape[0])) - winner_neuron[0], 2) / s
        gauss_x = np.power(np.asarray(range(self.neurons_map.shape[1])) - winner_neuron[1], 2) / s
        return np.exp(-1 * np.outer(gauss_x, gauss_y))

    def __can_grow(self, input_dataset):
        MQE = 0.0
        mapped_neurons = 0

        self.__map_data_to_neurons(input_dataset)
        neuron_iter = np.nditer(self.neurons_map, flags=['multi_index'])
        while not neuron_iter.finished:
            neuron = self.neurons_map[neuron_iter.multi_index]
            if len(neuron.input_dataset) != 0:
                MQE += neuron.compute_quantization_error()
                mapped_neurons += 1
            neuron_iter.iternext()

        return (MQE / mapped_neurons) >= (self.__t1 * self.__parent_quantization_error)

    def __map_data_to_neurons(self, input_dataset):
        # reset previous mapping
        neuron_iter = np.nditer(self.neurons_map, flags=['multi_index'])
        while not neuron_iter.finished:
            self.neurons_map[neuron_iter.multi_index].input_dataset.clear()
            neuron_iter.iternext()
        # finding the new association for each neuron
        for data in input_dataset:
            winner = self.winner_idx(data)
            self.neurons_map[winner].input_dataset.append(data)

    def __find_error_neuron(self, input_dataset):
        self.__map_data_to_neurons(input_dataset)

        quantization_errors_map = np.zeros(shape=self.neurons_map.shape, dtype=np.float32)
        qem_iter = np.nditer(quantization_errors_map, flags=['multi_index'])
        while not qem_iter.finished:
            neuron = self.neurons_map[qem_iter.multi_index]
            if len(neuron.input_dataset) != 0:
                quantization_errors_map[qem_iter.multi_index] += neuron.compute_quantization_error()
            qem_iter.iternext()

        return np.unravel_index(quantization_errors_map.max(), dims=quantization_errors_map.shape)

    def find_most_dissimilar_unit(self, error_unit_position):
        weight_distances_map = np.zeros(shape=self.neurons_map.shape, dtype=np.float32)

        error_neuron = self.neurons_map[error_unit_position]
        neuron_iter = np.nditer(self.neurons_map, flags=['multi_index'])
        while not neuron_iter.finished:
            if np.linalg.norm(error_unit_position - neuron_iter.multi_index, ord=1) == 1:
                neighbour = self.neurons_map[neuron_iter.multi_index]
                weight_distances_map[neuron_iter.multi_index] = error_neuron.weight_distance_from_other_unit(neighbour)
            neuron_iter.iternext()

        return np.unravel_index(weight_distances_map.max(), dims=weight_distances_map.shape)

    def grow(self):
        raise NotImplementedError

    def expand_column(self, error_unit_column_idx, dissimilar_unit_column_idx):
        raise NotImplementedError

    def expand_row(self, error_unit_row_idx, dissimilar_unit_row_idx):
        raise NotImplementedError

    def adjacent_units_direction(self, unit1, unit2):
        # returns horizontal or vertical
        raise NotImplementedError

    def init_new_unit_weight_vector(self, unit_idx):
        # the new units weight vector are initialized
        # as the average of their corresponding neighbors.
        raise NotImplementedError
