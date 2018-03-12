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
                self.neurons_map[position] = self.__build_neuron(weight)

    def __build_neuron(self, weight):
        # NOTE: reviewed
        return Neuron(weight, self.__parent_quantization_error, self.__t2, self.__growing_metric)

    def winner_idx(self, data):
        # NOTE: reviewed
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
            self.__update_neurons(data, lr, sigma)
            # updating lr and sigma
            lr *= decay
            sigma *= decay
            # updating map dimensions
            if (iter % epochs) == 0:
                can_grow = self.__can_grow(input_dataset)
                if can_grow:
                    self.grow(input_dataset)
                lr = learning_rate
                sigma = gaussian_sigma
            iter += 1

        print(self.neurons_map.shape)

        neurons_to_expand = list()
        self.__map_data_to_neurons(input_dataset)
        map_iter = np.nditer(self.neurons_map, flags=['multi_index', 'refs_ok'])
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

    def __update_neurons(self, data, learning_rate, sigma):
        gauss_kernel = self.__gaussian_kernel(self.winner_idx(data), sigma)
        # updating neurons weight
        map_iter = np.nditer(self.neurons_map, flags=['multi_index', 'refs_ok'])
        while not map_iter.finished:
            weight = self.neurons_map[map_iter.multi_index].weight_vector
            weight += learning_rate * gauss_kernel[map_iter.multi_index] * (data - weight)
            weight /= np.linalg.norm(weight)
            self.neurons_map[map_iter.multi_index].weight_vector = weight
            map_iter.iternext()

    def __gaussian_kernel(self, winner_neuron, gaussian_sigma):
        # TODO: kernel area != 1 (multiply kernel and A = 1/[2*pi*(gaussian_sigma**2)] to obtain a unit area). probably it's not necessary
        # computing gaussian kernel
        s = 2 * gaussian_sigma ** 2
        gauss_x = np.power(np.asarray(range(self.neurons_map.shape[0])) - winner_neuron[0], 2) / s
        gauss_y = np.power(np.asarray(range(self.neurons_map.shape[1])) - winner_neuron[1], 2) / s
        return np.exp(-1 * np.outer(gauss_x, gauss_y))

    def __can_grow(self, input_dataset):
        # NOTE: reviewed
        self.__map_data_to_neurons(input_dataset)

        MQE = 0.0
        mapped_neurons = 0
        for neuron in np.nditer(self.neurons_map):
            if len(neuron.input_dataset) != 0:
                MQE += neuron.compute_quantization_error()
                mapped_neurons += 1

        return (MQE / mapped_neurons) >= (self.__t1 * self.__parent_quantization_error)

    def __map_data_to_neurons(self, input_dataset):
        # NOTE: reviewed
        self.__clear_neurons_dataset()

        # finding the new association for each neuron
        for data in input_dataset:
            winner = self.winner_idx(data)
            self.neurons_map[winner].input_dataset.append(data)

    def __clear_neurons_dataset(self):
        # NOTE: reviewed
        for neuron in np.nditer(self.neurons_map):
            neuron.input_dataset.clear()

    def __find_error_neuron_idx(self, input_dataset):
        # NOTE: reviewed
        self.__map_data_to_neurons(input_dataset)

        quantization_errors = list()
        for neuron in np.nditer(self.neurons_map):
            quantization_error = -np.inf
            if len(neuron.input_dataset) != 0:
                quantization_error = neuron.compute_quantization_error()
            quantization_errors.append(quantization_error)

        return np.unravel_index(np.argmax(quantization_errors), dims=self.neurons_map.shape)

    def __find_most_dissimilar_neuron_idx(self, error_unit_position):
        # NOTE: reviewed
        error_neuron = self.neurons_map[error_unit_position]
        weight_distances = dict()

        neuron_iter = np.nditer(self.neurons_map, flags=['multi_index'])
        while not neuron_iter.finished:
            neuron_position = neuron_iter.multi_index
            if self.are_neurons_neighbours(error_unit_position, neuron_position):
                neighbour = self.neurons_map[neuron_position]
                weight_distances[neuron_position] = error_neuron.weight_distance_from_other_unit(neighbour)

            neuron_iter.iternext()

        return max(weight_distances, key=weight_distances.get)

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

            weight = np.zeros(shape=self.neurons_map[parent_position].weight_vector.shape, dtype=np.float32)
            for inner_idx in range(4):
                inner_pos = np.unravel_index(idx, dims=(2, 2))
                inner_pos = (__pos[0] + inner_pos[0], __pos[1] + inner_pos[1])
                if (inner_pos[0] >= 0) and (inner_pos[1] >= 0):
                    if (inner_pos[0] < self.neurons_map.shape[0]) and (inner_pos[1] < self.neurons_map.shape[1]):
                        weight += ( 0.25 * self.neurons_map[inner_pos].weight_vector )

            weights[position] = weight / np.linalg.norm(weight)

        return weights

    def grow(self, input_dataset):
        # NOTE: reviewed
        error_neuron_pos = self.__find_error_neuron_idx(input_dataset)
        dissimilar_neuron_pos = self.__find_most_dissimilar_neuron_idx(error_neuron_pos)

        if self.are_in_same_row(error_neuron_pos, dissimilar_neuron_pos):
            new_neuron_idxs = self.add_column_between(error_neuron_pos[1], dissimilar_neuron_pos[1])
            self.__init_new_neurons_weight_vector(new_neuron_idxs, "horizontal")

        elif self.are_in_same_column(error_neuron_pos, dissimilar_neuron_pos):
            new_neuron_idxs = self.add_row_between(error_neuron_pos[0], dissimilar_neuron_pos[0])
            self.__init_new_neurons_weight_vector(new_neuron_idxs, "vertical")
        else:
            raise RuntimeError("Error neuron and the most dissimilar are not adjacent")

    def add_column_between(self, error_unit_column_idx, dissimilar_unit_column_idx):
        # NOTE: reviewed
        line_size = self.neurons_map.shape[1]
        line = np.zeros(shape=line_size, dtype=object)
        new_column_idx = max(error_unit_column_idx, dissimilar_unit_column_idx)
        self.neurons_map = np.insert(self.neurons_map, new_column_idx, line, axis=1)

        return np.transpose(np.where(self.neurons_map == 0))

    def add_row_between(self, error_unit_row_idx, dissimilar_unit_row_idx):
        # NOTE: reviewed
        line_size = self.neurons_map.shape[0]
        line = np.zeros(shape=line_size, dtype=object)
        new_row_idx = max(error_unit_row_idx, dissimilar_unit_row_idx)
        self.neurons_map = np.insert(self.neurons_map, new_row_idx, line, axis=0)

        return np.transpose(np.where(self.neurons_map == 0))

    def __init_new_neurons_weight_vector(self, new_neuron_idxs, new_line_direction):
        # NOTE: reviewed
        for row, col in new_neuron_idxs:
            adjacent_neuron_idxs = self.__get_adjacent_neuron_idxs_by_direction(row, col, new_line_direction)
            weight_vector = self.__mean_weight_vector(adjacent_neuron_idxs)

            self.neurons_map[row, col] = self.__build_neuron(weight_vector)

    def __mean_weight_vector(self, neuron_idxs):
        # NOTE: reviewed
        weight_vector = np.zeros(shape=self.neurons_map[0, 0].weight_vector.shape, dtype=np.float32)
        for adjacent_idx in neuron_idxs:
            weight_vector += 0.5 * self.neurons_map[adjacent_idx].weight_vector
        return weight_vector

    @staticmethod
    def __get_adjacent_neuron_idxs_by_direction(row, col, direction):
        # NOTE: reviewed
        adjacent_neuron_idxs = list()
        if direction == "vertical":
            adjacent_neuron_idxs = [(row, col - 1), (row, col + 1)]

        elif direction == "horizontal":
            adjacent_neuron_idxs = [(row - 1, col), (row + 1, col)]

        return adjacent_neuron_idxs

    @staticmethod
    def are_neurons_neighbours(first_neuron_idx, second_neuron_idx):
        # NOTE: reviewed
        return np.linalg.norm(np.asarray(first_neuron_idx) - np.asarray(second_neuron_idx), ord=1) == 1

    @staticmethod
    def are_in_same_row(first_neuron_idx, second_neuron_idx):
        # NOTE: reviewed
        return abs(first_neuron_idx[0] - second_neuron_idx[0]) == 0

    @staticmethod
    def are_in_same_column(first_neuron_idx, second_neuron_idx):
        # NOTE: reviewed
        return abs(first_neuron_idx[1] - second_neuron_idx[1]) == 0
