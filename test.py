import numpy as np
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from collections import OrderedDict
from GHSOM import GHSOM
from matplotlib import pyplot as plt

data_shape = 28


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


def number_of_clusters(root_map):
    count = 0
    for neuron in root_map.neurons.values():
        if neuron.child_map is not None:
            count += number_of_clusters(neuron.child_map)
        else:
            count += 1
    return count


def used_units(root, dataset, labels):
    used_units = dict()
    for data, label in zip(dataset, labels):
        neuron = root
        while neuron.child_map is not None:
            map = neuron.child_map
            neuron = map.winner_neuron(data)[0][0]

        if neuron not in used_units:
            used_units[neuron] = dict()
        if label not in used_units[neuron]:
            used_units[neuron][label] = 0
        used_units[neuron][label] += 1

    return used_units


def depth(map, init):
    depths = [init]
    for neuron in map.neurons.values():
        if neuron.child_map is not None:
            depths.append(depth(neuron.child_map, init + 1))
    return max(depths)


def _max_size(map):
    sizes = {len(map.neurons): map.weights_map[0].shape[0:2]}
    for neuron in map.neurons.values():
        if neuron.child_map is not None:
            sizes.update(_max_size(neuron.child_map))

    return sizes


def max_size(map):
    sizes = _max_size(map)
    size = max(sizes.keys())
    return (size, sizes[size])


def sizes(map):
    s = [map.weights_map[0].shape[0:2]]
    for neuron in map.neurons.values():
        if neuron.child_map is not None:
            s.extend(sizes(neuron.child_map))

    return s


def mean_size(map):
    return np.asarray(sizes(map)).mean(axis=0)


def number_of_map(root):
    count = 0
    if root.child_map is not None:
        count = 1
        for unit in root.child_map.neurons.values():
            count += number_of_map(unit)
    return count


if __name__ == '__main__':
    # dataset = load_digits()
    dataset = fetch_mldata('MNIST original')

    # gen = np.random.RandomState(0)
    # rand_idxs = gen.randint(dataset.data.shape[0], size=20000)

    # data = np.take(dataset.data, rand_idxs, axis=0)
    data = dataset.data
    n_samples, n_features = data.shape
    n_digits = len(np.unique(dataset.target))
    # labels = np.take(dataset.target, rand_idxs).astype(int)
    labels = dataset.target.astype(int)

    t1 = 0.1
    t2 = 0.0001
    learning_rate = 0.15
    decay = 0.95
    gaussian_sigma = 1.5

    ghsom = GHSOM(input_dataset=data, t1=t1, t2=t2, learning_rate=learning_rate, decay=decay, gaussian_sigma=gaussian_sigma)
    print("Training...")
    zero_unit = ghsom.train(epochs_number=15, dataset_percentage=0.30, min_dataset_size=30, seed=0, grow_maxiter=np.inf)

    print("#"*5, "Dataset details", "#"*5)
    print("dataset length: {}".format(n_samples))
    print("features per example: {}".format(n_features))
    print("number of digits: {}\n".format(n_digits))

    print("#"*5, "Parameters", "#"*5)
    print("Tau 1: ", t1)
    print("Tau 2: ", t2)
    print("Learning Rate: ", learning_rate)
    print("Sigma: ", gaussian_sigma)
    print("Decay: ", decay)

    print("#"*5, "Metrics", "#"*5)
    units = used_units(zero_unit, data, labels)
    print("Tree height: ", depth(zero_unit.child_map, 0))
    print("Mean map size: ", mean_size(zero_unit.child_map))
    print("Max map size: ", max_size(zero_unit.child_map))
    print("Number of maps: ", number_of_map(zero_unit))
    print("Number of clusters: ", number_of_clusters(zero_unit.child_map))
    print("Number of used clusters: ", len(units))
    print("Number of non-homogeneous clusters: ", len(list(filter(lambda x: len(x) > 1, units.values()))))

    #interactive_plot(zero_unit.child_map, data)
    # print(zero_unit)
    # print(mean_data_centroid_activation(zero_unit, data))
    # print(dispersion_rate(zero_unit, data))
    # interactive_plot_with_labels(zero_unit.child_map, data, labels)
    # plt.show()
