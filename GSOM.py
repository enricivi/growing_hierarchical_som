class GSOM:
    def __init__(self, initial_map_size, parent_quantization_error, t1, weight_vectors_dict=None):
        self.__t1 = t1
        self.__parent_quantization_error = parent_quantization_error
        self.__map_size = initial_map_size

        if weight_vectors_dict is not None:
            # init weight vectors
            raise NotImplementedError

    def winner_idx(self):
        raise NotImplementedError

    def train(self, input_dataset, epochs, learning_rate, decay, gaussian_sigma, t2):
        raise NotImplementedError

    def can_grow(self):
        # MQE
        raise NotImplementedError

    def find_error_unit(self):
        raise NotImplementedError

    def find_most_dissimilar_unit(self, error_unit):
        raise NotImplementedError

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
