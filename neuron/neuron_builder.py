from neuron.neuron import Neuron


class NeuronBuilder:
    class __SingletonNeuronBuilder:
        def __init__(self, tau_2, growing_metric):
            self.zero_quantization_error = None
            self.growing_metric = growing_metric
            self.tau_2 = tau_2

        def set_zero_quantization_error(self, zero_quantization_error):
            self.zero_quantization_error = zero_quantization_error

    instance = None

    @staticmethod
    def new_neuron(weights_map, position):
        assert NeuronBuilder.instance is not None, "zero_neuron not initialized"

        return Neuron(
            weights_map,
            position,
            NeuronBuilder.instance.zero_quantization_error,
            NeuronBuilder.instance.tau_2,
            NeuronBuilder.instance.growing_metric
        )

    @staticmethod
    def zero_neuron(weight_vector, input_dataset, tau_2, growing_metric):
        if NeuronBuilder.instance is None:
            NeuronBuilder.instance = NeuronBuilder.__SingletonNeuronBuilder(tau_2, growing_metric)

        zero_neuron = Neuron([weight_vector.reshape(1, 1, input_dataset.shape[1])], (0, 0), None, None,
                             NeuronBuilder.instance.growing_metric)
        zero_neuron.input_dataset = input_dataset

        NeuronBuilder.instance.set_zero_quantization_error(zero_neuron.compute_quantization_error())

        return zero_neuron

    @staticmethod
    def get_zero_quantization_error():
        assert NeuronBuilder.instance is not None, "zero_neuron not initialized"

        return NeuronBuilder.instance.zero_quantization_error
