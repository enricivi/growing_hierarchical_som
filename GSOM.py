from neuron import NeuronBuilder
import numpy as np


class GSOM:
    def __init__(self, initial_map_size, parent_quantization_error, t1, t2, growing_metric, data_size,
                 weights_map, parent_dataset):
        assert parent_dataset is not None, "Provided dataset is empty"
        self.__data_size = data_size
        self.__t1 = t1
        self.__t2 = t2
        self.__parent_quantization_error = parent_quantization_error
        self.__initial_map_size = initial_map_size
        self.__growing_metric = growing_metric
        self.__parent_dataset = parent_dataset

        self.weights_map = weights_map

        self.neurons = self.__build_neurons_list()

    def winner_neuron(self, data):
        # NOTE: reviewed
        activations = list()
        for neuron in self.neurons.values():
            activations.append(neuron.activation(data))

        idx = np.unravel_index(np.argmin(activations), dims=self.__map_shape())
        return self.neurons[idx]

    def train(self, epochs, initial_gaussian_sigma, initial_learning_rate, decay):
        can_grow = True
        while can_grow:
            self.__neurons_training(decay, epochs, self.__parent_dataset, initial_learning_rate, initial_gaussian_sigma)

            can_grow = self.__can_grow(self.__parent_dataset)
            if can_grow:
                self.grow(self.__parent_dataset)

        self.__map_data_to_neurons(self.__parent_dataset)

    def __neurons_training(self, decay, epochs, input_dataset, learning_rate, sigma):
        for iteration in range(epochs):
            data = input_dataset[np.random.randint(len(input_dataset))]
            self.__update_neurons(data, learning_rate, sigma)

            learning_rate *= decay
            sigma *= decay

    def __update_neurons(self, data, learning_rate, sigma):
        gauss_kernel = self.__gaussian_kernel(self.winner_neuron(data), sigma)

        for neuron in self.neurons.values():
            weight = neuron.weight_vector()
            weight += learning_rate * gauss_kernel[neuron.position] * (data - weight)
            weight /= np.linalg.norm(weight)
            self.weights_map[neuron.position] = weight

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
        winner_row, winner_col = winner_neuron.position
        s = 2 * gaussian_sigma ** 2

        gauss_col = np.power(np.asarray(range(self.__map_shape()[1])) - winner_col, 2) / s
        gauss_row = np.power(np.asarray(range(self.__map_shape()[0])) - winner_row, 2) / s

        return np.exp(-1 * np.outer(gauss_col, gauss_row))

    def __can_grow(self, input_dataset):
        # NOTE: reviewed
        self.__map_data_to_neurons(input_dataset)

        MQE = 0.0
        mapped_neurons = 0
        for neuron in self.neurons.values():
            if neuron.input_dataset is not None:
                MQE += neuron.compute_quantization_error()
                mapped_neurons += 1

        return (MQE / mapped_neurons) >= (self.__t1 * self.__parent_quantization_error)

    def __map_data_to_neurons(self, input_dataset):
        # NOTE: reviewed
        self.__clear_neurons_dataset()

        # finding the new association for each neuron
        for data in input_dataset:
            winner = self.winner_neuron(data)
            winner.input_dataset.append(data)

    def __clear_neurons_dataset(self):
        # NOTE: reviewed
        for neuron in self.neurons.values():
            neuron.input_dataset.clear()

    def __find_error_neuron(self, input_dataset):
        # NOTE: reviewed
        self.__map_data_to_neurons(input_dataset)

        quantization_errors = list()
        for neuron in self.neurons.values():
            quantization_error = -np.inf
            if neuron.input_dataset is not None:
                quantization_error = neuron.compute_quantization_error()
            quantization_errors.append(quantization_error)

        idx = np.unravel_index(np.argmax(quantization_errors), dims=self.__map_shape())
        return self.neurons[idx]

    def __find_most_dissimilar_neuron(self, error_neuron):
        # NOTE: reviewed
        weight_distances = dict()
        for neuron in self.neurons.values():
            if self.are_neurons_neighbours(error_neuron, neuron):
                weight_distances[neuron] = error_neuron.weight_distance_from_other_unit(neuron)

        return max(weight_distances, key=weight_distances.get)

    def grow(self, input_dataset):
        # NOTE: reviewed
        error_neuron = self.__find_error_neuron(input_dataset)
        dissimilar_neuron = self.__find_most_dissimilar_neuron(error_neuron)

        if self.are_in_same_row(error_neuron, dissimilar_neuron):
            new_neuron_idxs = self.add_column_between(error_neuron, dissimilar_neuron)
            self.__init_new_neurons_weight_vector(new_neuron_idxs, "horizontal")

        elif self.are_in_same_column(error_neuron, dissimilar_neuron):
            new_neuron_idxs = self.add_row_between(error_neuron, dissimilar_neuron)
            self.__init_new_neurons_weight_vector(new_neuron_idxs, "vertical")
        else:
            raise RuntimeError("Error neuron and the most dissimilar are not adjacent")

    def add_column_between(self, error_neuron, dissimilar_neuron):
        # NOTE: reviewed
        error_col = error_neuron.position[1]
        dissimilar_col = dissimilar_neuron.position[1]
        new_column_idx = max(error_col, dissimilar_col)

        line_size = self.__map_shape()[0]
        line = np.zeros(shape=line_size, dtype=np.float32)
        self.weights_map = np.insert(self.weights_map, new_column_idx, line, axis=1)

        return np.transpose(np.where(self.weights_map == 0))

    def add_row_between(self, error_neuron, dissimilar_neuron):
        # NOTE: reviewed
        error_row = error_neuron.position[0]
        dissimilar_row = dissimilar_neuron.position[0]
        new_row_idx = max(error_row, dissimilar_row)

        line_size = self.__map_shape()[1]
        line = np.zeros(shape=line_size, dtype=np.float32)
        self.weights_map = np.insert(self.weights_map, new_row_idx, line, axis=0)

        return np.transpose(np.where(self.weights_map == 0))

    def __init_new_neurons_weight_vector(self, new_neuron_idxs, new_line_direction):
        # NOTE: reviewed
        for row, col in new_neuron_idxs:
            adjacent_neuron_idxs = self.__get_adjacent_neuron_idxs_by_direction(row, col, new_line_direction)
            weight_vector = self.__mean_weight_vector(adjacent_neuron_idxs)

            self.weights_map[row, col] = weight_vector
            self.neurons[(row, col)] = self.__build_neuron((row, col))

    def __mean_weight_vector(self, neuron_idxs):
        # NOTE: reviewed
        weight_vector = np.zeros(shape=self.__data_size, dtype=np.float32)
        for adjacent_idx in neuron_idxs:
            weight_vector += 0.5 * self.neurons[adjacent_idx].weight_vector()
        return weight_vector / np.linalg.norm(weight_vector)

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
    def are_neurons_neighbours(first_neuron, second_neuron):
        # NOTE: reviewed
        return np.linalg.norm(np.asarray(first_neuron.position) - np.asarray(second_neuron.position), ord=1) == 1

    @staticmethod
    def are_in_same_row(first_neuron, second_neuron):
        # NOTE: reviewed
        return abs(first_neuron.position[0] - second_neuron.position[0]) == 0

    @staticmethod
    def are_in_same_column(first_neuron, second_neuron):
        # NOTE: reviewed
        return abs(first_neuron.position[1] - second_neuron.position[1]) == 0

    def __build_neurons_list(self):
        rows, cols = self.__initial_map_size
        return {(x, y): self.__build_neuron((x, y)) for x in range(rows) for y in range(cols)}

    def __build_neuron(self, weight_position):
        # NOTE: reviewed
        return NeuronBuilder.new_neuron(weight_position, self.weights_map)

    def __map_shape(self):
        shape = self.weights_map.shape
        return shape[0], shape[1]
