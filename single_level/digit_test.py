from single_level.self_organizing_map import SOM

import numpy as np
from sklearn.datasets import load_digits
from collections import OrderedDict
from time import time
import matplotlib.pyplot as plt


def print_symbols_heat_map(_som, _digits):
    plt.figure()
    plt.imshow(_som.heat_map(digits.data), cmap='bone')
    markers = ['o', 'v', '1', '3', '8', 's', 'p', 'x', 'D', '*']
    colors = ["r", "g", "b", "y", "c", (0, 0.1, 0.8), (1, 0.5, 0), (1, 1, 0.3), "m", (0.4, 0.6, 0)]
    m_size = 192 / _som.map_dim['w']
    m_width = 32 / _som.map_dim['w']
    for _data, _label in zip(_digits.data, _digits.target):
        marker = markers[_label]
        color = colors[_label]
        winner_idx = _som.min_activation_idx(_data)
        plt.plot(winner_idx[1], winner_idx[0], marker, markerfacecolor='None',
                 markeredgecolor=color, markersize=m_size, markeredgewidth=m_width, label=_label)
    legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(legend_labels, legend_handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0., 1.02, 1., .102), ncol=5,
               loc=3, borderaxespad=0., mode="expand")
    plt.colorbar()


def print_full_map(_som):
    plt.figure()
    map_width = 8 * _som.map_dim['w']
    map_height = 8 * _som.map_dim['h']
    full_map = np.empty(shape=(map_width, map_height), dtype=np.float32)
    for i in range(0, map_width, 8):
        for j in range(0, map_height, 8):
            neuron = _som.map[i // 8, j // 8]
            full_map[i:(i + 8), j:(j + 8)] = np.reshape(neuron, newshape=(8, 8))
    plt.imshow(full_map, cmap='bone_r', interpolation='sinc')


digits = load_digits()

data = digits.data
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

print("dataset length: {}".format(n_samples))
print("features per example: {}".format(n_features))
print("number of digits: {}".format(n_digits))

m = {'ed': None, '1d': lambda x, y: np.linalg.norm(x - y, ord=1), 'inf': lambda x, y: np.linalg.norm(x - y, ord=np.inf)}
som = SOM(map_dim=(16, 16, n_features), metrics=m['ed'], learning_rate=0.5, update_mask_dim=(5, 5), sigma=1.0,
          decay=0.99)

plt.figure(num='gaussian kernel (update_mask)')
plt.imshow(som.update_mask, cmap='bone')
plt.colorbar()

som.rand_init_from_data(dataset=data, seed=1)

print_symbols_heat_map(som, digits)
print_full_map(som)

start = time()
som.train(dataset=data, epochs=10, seed=2)
print("elapsed time: {:.1f} seconds".format(time() - start))

print_symbols_heat_map(som, digits)
print_full_map(som)
plt.show()
