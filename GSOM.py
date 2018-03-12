from neuron import Neuron
import numpy as np


class GSOM:
    def __init__(self, initial_map_size, parent_quantization_error, t1, t2, growing_metric, weights_vectors_dict=None):
        self.__t1 = t1
        self.__t2 = t2
        self.__parent_quantization_error = parent_quantization_error
        self.__init_map_size = initial_map_size
        self.__growing_metric = growing_metric

        self.neurons_map = np.zeros(initial_map_size, dtype=object)

        if weights_vectors_dict is not None:
            for position, weight in weights_vectors_dict.items():
                self.neurons_map[position] = self.__build_neuron(weight, growing_metric)

    def __build_neuron(self, weight, growing_metric):
        return Neuron(weight, self.__parent_quantization_error, self.__t2, growing_metric)

    def winner_idx(self, data):
        activations = list()
        for neuron in np.nditer(self.neurons_map):
            activations.append(neuron.activation(data))

        return np.unravel_index(np.argmin(activations), dims=self.neurons_map.shape)

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

        neurons_to_expand = list()
        self.__map_data_to_neurons(input_dataset)
        map_iter = np.nditer(self.neurons_map, flags=['multi_index'])
        while not map_iter.finished:
            if self.neurons_map[map_iter.multi_index].needs_child_map():
                neurons_to_expand.append(map_iter.multi_index)
            map_iter.iternext()
        for neuron_pos in neurons_to_expand:
            self.neurons_map[neuron_pos].child_map = GSOM(
                self.__init_map_size,
                self.neurons_map[neuron_pos].compute_quantization_error(),
                self.__t1,
                self.__t2,
                self.__growing_metric,
                self.__init_new_map(neuron_pos)
            )
            self.neurons_map[neuron_pos].child_map.train(
                np.asarray(self.neurons_map[neuron_pos].input_dataset),
                epochs, learning_rate, decay, gaussian_sigma
            )

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
        s = 2 * gaussian_sigma ** 2
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
        self.__clear_neurons_dataset()

        # finding the new association for each neuron
        for data in input_dataset:
            winner = self.winner_idx(data)
            self.neurons_map[winner].input_dataset.append(data)

    def __clear_neurons_dataset(self):
        for neuron in np.nditer(self.neurons_map):
            neuron.input_dataset.clear()

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

    def __find_most_dissimilar_neuron(self, error_unit_position):
        weight_distances_map = np.zeros(shape=self.neurons_map.shape, dtype=np.float32)

        error_neuron = self.neurons_map[error_unit_position]
        neuron_iter = np.nditer(self.neurons_map, flags=['multi_index'])
        while not neuron_iter.finished:
            if np.linalg.norm(error_unit_position - neuron_iter.multi_index, ord=1) == 1:
                neighbour = self.neurons_map[neuron_iter.multi_index]
                weight_distances_map[neuron_iter.multi_index] = error_neuron.weight_distance_from_other_unit(neighbour)
            neuron_iter.iternext()

        return np.unravel_index(weight_distances_map.max(), dims=weight_distances_map.shape)

    def __init_new_map(self, parent_position):
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
        # TODO: this method must be generalized
        parent_neighbourhood_first_position = (parent_position[0]-1, parent_position[1]-1)
        weights = dict()
        for idx in range(4):
            position = np.unravel_index(idx, dims=(2, 2))
            __pos = (
                parent_neighbourhood_first_position[0] + position[0],
                parent_neighbourhood_first_position[1] + position[1]
            )

            weight = np.zeros(shape=self.neurons_map[parent_position].__weight_vector.shape, dtype=np.float32)
            for inner_idx in range(4):
                inner_pos = np.unravel_index(idx, dims=(2, 2))
                inner_pos = (__pos[0] + inner_pos[0], __pos[1] + inner_pos[1])
                if (inner_pos[0] >= 0) and (inner_pos[1] >= 0):
                    if (inner_pos[0] < self.neurons_map.shape[0]) and (inner_pos[1] < self.neurons_map.shape[1]):
                        weight += ( 0.25 * self.neurons_map[inner_pos] )

            weights[position] = weight / np.linalg.norm(weight)

        return weights

    def grow(self, input_dataset):
        error_neuron_pos = self.__find_error_neuron(input_dataset)
        dissimilar_neuron_pos = self.__find_most_dissimilar_neuron(error_neuron_pos)    # strange error
        if (dissimilar_neuron_pos[0] - error_neuron_pos[0]) == 0:
            self.expand_column(error_neuron_pos[1], dissimilar_neuron_pos[1])
        else:
            self.expand_row(error_neuron_pos[0], dissimilar_neuron_pos[0])
        self.__init_new_neurons_weight_vector()

    def expand_column(self, error_unit_column_idx, dissimilar_unit_column_idx):
        new_neurons = self.neurons_map.shape[1]
        neurons = np.zeros(shape=(1, new_neurons), dtype=object)
        np.insert(self.neurons_map, max(error_unit_column_idx, dissimilar_unit_column_idx), neurons, axis=1)

    def expand_row(self, error_unit_row_idx, dissimilar_unit_row_idx):
        new_neurons = self.neurons_map.shape[0]
        neurons = np.zeros(shape=(1, new_neurons), dtype=object)
        np.insert(self.neurons_map, max(error_unit_row_idx, dissimilar_unit_row_idx), neurons, axis=0)

    def __init_new_neurons_weight_vector(self):
        new_neurons = np.where(self.neurons_map == 0)
        axis = ((0, 0), (1, -1)) if (np.mean(new_neurons[0] == new_neurons[0][0])) else ((1, -1), (0, 0))
        for i, j in new_neurons:
            weight = np.zeros(shape=self.neurons_map[0, 0].__weight_vector.shape, dtype=np.float32)
            for di, dj in axis:
                weight += 0.5 * self.neurons_map[i + di, j + dj].__weight_vector
            self.neurons_map[i, j].__weight_vector = weight / np.linalg.norm(weight)
