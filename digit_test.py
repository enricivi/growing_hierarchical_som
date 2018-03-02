from self_organizing_map import SOM

import numpy as np
from sklearn.datasets import load_digits
from time import time
import matplotlib.pyplot as plt


def print_symbols_heat_map(som, digits):
    plt.figure()
    plt.imshow(som.heat_map(digits.data), cmap='bone')
    markers = ['o', 'v', '1', '3', '8', 's', 'p', 'x', 'D', '*']
    colors = ["r", "g", "b", "y", "c", (0, 0.1, 0.8), (1, 0.5, 0), (1, 1, 0.3), "m", (0.4, 0.6, 0)]
    for data, label in zip(digits.data, digits.target):
        marker = markers[label]
        color = colors[label]
        winnerIdx = som.min_activation_idx(data)
        plt.plot(winnerIdx[0], winnerIdx[1], marker, markerfacecolor='None',
                 markeredgecolor=color, markersize=12, markeredgewidth=2)


digits = load_digits()

data = digits.data
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

print( "dataset length: {}".format(n_samples) )
print( "features per example: {}".format(n_features) )
print( "number of digits: {}".format(n_digits) )

pos = (0, 0)
m = {'ed': None, '1d': lambda x, y: np.linalg.norm(x-y, ord=1), 'inf': lambda x, y: np.linalg.norm(x-y, ord=np.inf)}

som = SOM(map_dim=(16, 16, n_features), metrics=m['ed'], learning_rate=0.5, update_mask_dim=(5, 5), sigma=1.0, decay=0.99)

plt.figure(num='gaussian kernel (update mask)')
plt.imshow(som.update_mask, cmap='bone')

plt.figure(num='map position {} - pre init'.format(pos))
plt.imshow(np.reshape(som.map[pos], (8, 8)), cmap='bone')

som.rand_init_from_data(data=data, seed=1)

plt.figure(num='map position {} - after init'.format(pos))
plt.imshow(np.reshape(som.map[pos], (8, 8)), cmap='bone')

plt.figure(num='heat map - pre train')
plt.imshow(som.heat_map(data), cmap='bone')

print_symbols_heat_map(som, digits)
start = time()
som.train(data=data, epochs=1, seed=2)
print( "elpased time: {:.2f} seconds".format(time()-start) )

print_symbols_heat_map(som, digits)

plt.figure(num='heat map - after training')
plt.imshow(som.heat_map(data), cmap='bone')

plt.figure(num='map position {} - after train'.format(pos))
plt.imshow(np.reshape(som.map[pos], (8, 8)), cmap='bone')
plt.show()
