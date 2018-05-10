import numpy as np
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from collections import OrderedDict
from GHSOM import GHSOM
from matplotlib import pyplot as plt

data_shape = 8


def __gmap_to_matrix(gmap):
    gmap = gmap[0]
    map_row = data_shape * gmap.shape[0]
    map_col = data_shape * gmap.shape[1]
    _image = np.empty(shape=(map_row, map_col), dtype=np.float32)
    for i in range(0, map_row, data_shape):
        for j in range(0, map_col, data_shape):
            neuron = gmap[i // data_shape, j // data_shape]
            _image[i:(i + data_shape), j:(j + data_shape)] = np.reshape(neuron, newshape=(data_shape, data_shape))
    return _image


def __plot_child(e, gmap, level):
    if e.inaxes is not None:
        coords = (int(e.ydata // data_shape), int(e.xdata // data_shape))
        neuron = gmap.neurons[coords]
        if neuron.child_map is not None:
            interactive_plot(neuron.child_map, num=str(coords), level=level+1)


def interactive_plot(gmap, num='root', level=1):
    _num = "level {} -- parent pos {}".format(level, num)
    fig, ax = plt.subplots(num=_num)
    ax.imshow(__gmap_to_matrix(gmap.weights_map), cmap='bone_r', interpolation='sinc')
    fig.canvas.mpl_connect('button_press_event', lambda event: __plot_child(event, gmap, level))
    plt.axis('off')
    fig.show()


def __plot_child_with_labels(e, gmap, level, data, labels, associations):
    if e.inaxes is not None:
        coords = (int(e.ydata // data_shape), int(e.xdata // data_shape))
        neuron = gmap.neurons[coords]
        if neuron.child_map is not None:
            assc = associations[coords[0]][coords[1]]
            interactive_plot_with_labels(neuron.child_map, dataset=data[assc], labels=labels[assc],
                                         num=str(coords), level=level+1)


def interactive_plot_with_labels(gmap, dataset, labels, num='root', level=1):
    colors = ["#E52B50", "#FFBF00", "#4B0082", "#FBCEB1", "#7FFFD4",
              "#007FFF", "#00FF00", "#9966CC", "#CD7F32", "#89CFF0"]
    sizes = np.arange(0, 60, 6) + 0.5

    mapping = [[list() for _ in range(gmap.map_shape()[1])] for _ in range(gmap.map_shape()[0])]

    _num = "level {} -- parent pos {}".format(level, num)
    fig, ax = plt.subplots(num=_num)
    ax.imshow(__gmap_to_matrix(gmap.weights_map), cmap='bone_r', interpolation='sinc')
    fig.canvas.mpl_connect('button_press_event', lambda event: __plot_child_with_labels(event, gmap, level,
                                                                                        dataset, labels, mapping))
    plt.axis('off')

    for idx, label in enumerate(labels):
        winner_neuron = gmap.winner_neuron(dataset[idx])[0][0]
        r, c = winner_neuron.position
        mapping[r][c].append(idx)

        ax.plot(c*data_shape+data_shape/2, r*data_shape+data_shape/2, 'o', markerfacecolor='None',
                markeredgecolor=colors[label], markersize=sizes[label], markeredgewidth=1.5, label=label)
    legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(legend_labels, legend_handles))
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.1, 0.5), borderaxespad=0.,
               mode='expand', labelspacing=int((gmap.map_shape()[0]/9)*data_shape))
    fig.show()


def mean_data_centroid_activation(ghsom, dataset):
    distances = list()

    for data in dataset:
        _neuron = ghsom
        while _neuron.child_map is not None:
            _gsom = _neuron.child_map
            _neuron = _gsom.winner_neuron(data)[0][0]
        distances.append(_neuron.activation(data))

    distances = np.asarray(a=distances, dtype=np.float32)
    return distances.mean(), distances.std()


def __number_of_neurons(root):
    r, c = root.child_map.weights_map[0].shape[0:2]
    total_neurons = r * c
    for neuron in root.child_map.neurons.values():
        if neuron.child_map is not None:
            total_neurons += __number_of_neurons(neuron)
    return total_neurons


def dispersion_rate(ghsom, dataset):
    used_neurons = dict()
    for data in dataset:
        gsom_reference = ''
        neuron_reference = ''
        _neuron = ghsom
        while _neuron.child_map is not None:
            _gsom = _neuron.child_map
            _neuron = _gsom.winner_neuron(data)[0][0]

            gsom_reference = str(_gsom)
            neuron_reference = str(_neuron)

        used_neurons["{}-{}-{}".format(gsom_reference, neuron_reference, _neuron.position)] = True
    used_neurons = len(used_neurons)

    return __number_of_neurons(ghsom) / used_neurons


if __name__ == '__main__':
    digits = load_digits()

    data = digits.data
    n_samples, n_features = data.shape
    n_digits = len(np.unique(digits.target))
    labels = digits.target

    # mnist = fetch_mldata('MNIST original')
    # data = mnist.data
    # n_samples, n_features = data.shape
    # n_digits = len(np.unique(mnist.target))
    # labels = mnist.target

    print("dataset length: {}".format(n_samples))
    print("features per example: {}".format(n_features))
    print("number of digits: {}\n".format(n_digits))
    ghsom = GHSOM(input_dataset=data, t1=0.1, t2=0.0001, learning_rate=0.15, decay=0.95, gaussian_sigma=1.5)

    print("Training...")
    zero_unit = ghsom.train(epochs_number=15, dataset_percentage=0.50, min_dataset_size=30, seed=0, grow_maxiter=10)

    print(zero_unit)
    print(mean_data_centroid_activation(zero_unit, data))
    print(dispersion_rate(zero_unit, data))
    interactive_plot_with_labels(zero_unit.child_map, data, labels)
    plt.show()
