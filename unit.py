class Unit:
    def __init__(self, weight_vector, input_dataset, zero_quantization_error, t2):
        """
        :type t2: The tau_2 parameter
        :type zero_quantization_error: The quantization error of the layer 0
        :type input_dataset: List of the inputs that map in this unit (maybe)
        :type weight_vector: the initial weight vector for the unit - m_i
        """
        self.__t2 = t2
        self.__zero_quantization_error = zero_quantization_error
        self.__input_dataset = input_dataset
        self.__weight_vector = weight_vector

        self.child_map = None

    def needs_child_map(self):
        # global stopping criterion
        raise NotImplementedError
        return False

    def calc_quantization_error(self):
        raise NotImplementedError

    def distance_from_other_unit(self, unit):
        # 2-norm
        raise NotImplementedError

    def __distance_from_input_data_item(self, data_item):
        # 2-norm
        raise NotImplementedError
