import numpy as np
from matplotlib import pyplot as plt


def explore(root_unit, map_per_level, level=1, parent=None):
    if not(level in map_per_level):
        map_per_level[level] = list()

    map_per_level[level].append((
        root_unit.child_map.weights_map[0],
        parent
    ))

    for neuron in root_unit.child_map.neurons.values():
        if neuron.child_map is not None:
            explore(neuron, map_per_level, level + 1, neuron)


def plot(map_per_level, level):
    weights = map_per_level[level]
    for idx, weight in enumerate(weights):
        weight = weight[0]
        plt.figure("Level {} - Weight {}".format(level, idx))
        map_row = 8 * weight.shape[0]
        map_col = 8 * weight.shape[1]
        image_weight = np.empty(shape=(map_row, map_col), dtype=np.float32)
        for i in range(0, map_row, 8):
            for j in range(0, map_col, 8):
                neuron = weight[i // 8, j // 8]
                image_weight[i:(i + 8), j:(j + 8)] = np.reshape(neuron, newshape=(8, 8))
        plt.imshow(image_weight, cmap='bone_r', interpolation='sinc')
    plt.show()
