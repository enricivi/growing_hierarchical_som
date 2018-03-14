from neuron import Neuron


class NeuronBuilder:
    def __init__(self, zero_quantization_error, tau_2, growing_metric):
        self.growing_metric = growing_metric
        self.tau_2 = tau_2
        self.zero_quantization_error = zero_quantization_error

    def new_neuron(self, weights_map, position):
        return Neuron(weights_map, position, self.zero_quantization_error, self.tau_2, self.growing_metric)

    def zero_unit(self, weight_vector, input_dataset):
        # finish
        pass