import numpy as np
from sklearn.datasets import load_digits
from time import time
from GHSOM import GHSOM
from matplotlib import pyplot as plt


def __gmap_to_matrix(gmap):
    gmap = gmap[0]
    map_row = 8 * gmap.shape[0]
    map_col = 8 * gmap.shape[1]
    _image = np.empty(shape=(map_row, map_col), dtype=np.float32)
    for i in range(0, map_row, 8):
        for j in range(0, map_col, 8):
            neuron = gmap[i // 8, j // 8]
            _image[i:(i + 8), j:(j + 8)] = np.reshape(neuron, newshape=(8, 8))
    return _image


def __plot_child(e, gmap, level):
    if e.inaxes is not None:
        coords = (int(e.ydata // 8), int(e.xdata // 8))
        neuron = gmap.neurons[coords]
        if neuron.child_map is not None:
            interactive_plot(neuron.child_map, num=str(coords), level=level+1)


def interactive_plot(gmap, num='root', level=0):
    _num = "level {} -- parent pos {}".format(level, num)
    fig, ax = plt.subplots(num=_num)
    ax.imshow(__gmap_to_matrix(gmap.weights_map), cmap='bone_r', interpolation='sinc')
    fig.canvas.mpl_connect('button_press_event', lambda event: __plot_child(event, gmap, level))
    plt.axis('off')
    fig.show()


def plot(gmap):
    plt.figure(num="parent")
    plt.imshow(__gmap_to_matrix(gmap.weights_map), cmap='bone_r', interpolation='sinc')
    for neuron in gmap.neurons.values():
        if neuron.child_map is not None:
            plt.figure(num=str(neuron.position))
            plt.imshow(__gmap_to_matrix(neuron.child_map.weights_map), cmap='bone_r', interpolation='sinc')
    plt.show()


def distances_test(dataset, zero_unit):
    distances = list()
    for data in dataset:
        _neuron = zero_unit
        while _neuron.child_map is not None:
            gsom = _neuron.child_map
            _neuron = gsom.winner_neuron(data=data)
        distances.append(np.linalg.norm(_neuron.weight_vector() - data))
    distances = np.asanyarray(distances, dtype=np.float32)
    return distances.mean(), distances.std(), max(distances), min(distances)


if __name__ == '__main__':
    digits = load_digits()

    data = digits.data
    n_samples, n_features = data.shape
    n_digits = len(np.unique(digits.target))
    labels = digits.target

    print("dataset length: {}".format(n_samples))
    print("features per example: {}".format(n_features))
    print("number of digits: {}\n".format(n_digits))

    ghsom = GHSOM(input_dataset=data, t1=0.25, t2=0.05, learning_rate=0.01, decay=0.9, gaussian_sigma=1)

    print("Training...")
    zero_unit = ghsom.train(epochs_number=30, dataset_percentage=0.25, min_dataset_size=70, seed=1, grow_maxiter=30)

    print(zero_unit)
    interactive_plot(zero_unit.child_map)

    print(distances_test(data, zero_unit))
