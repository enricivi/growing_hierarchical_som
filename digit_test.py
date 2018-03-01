from self_organizing_map import SOM

import numpy as np
from sklearn.datasets import load_digits
from time import time
import matplotlib.pyplot as plt

digits = load_digits()

data = digits.data
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

print( "dataset length: {}".format(n_samples) )
print( "features per example: {}".format(n_features) )
print( "number of digits: {}".format(n_digits) )

pos = (15, 15)
m = {'ed': None, '1d': lambda x, y: np.linalg.norm(x-y, ord=1), 'inf': lambda x, y: np.linalg.norm(x-y, ord=np.inf)}

som = SOM(map_dim=(16, 16, n_features), metrics=m['ed'], learning_rate=0.5, update_mask_dim=(5, 5), sigma=1.0, decay=0.99)

plt.figure(num='gaussian kernel (update mask)')
plt.imshow(som.update_mask, cmap='bone')

plt.figure(num='map position {} - pre init'.format(pos))
plt.imshow(np.reshape(som.map[pos], (8, 8)), cmap='bone')

som.rand_init_from_data(data=data, seed=1)

plt.figure(num='map position {} - after init'.format(pos))
plt.imshow(np.reshape(som.map[pos], (8, 8)), cmap='bone')

start = time()
som.train(data=data, epochs=10, seed=2)
print( "elpased time: {:.2f} seconds".format(time()-start) )

plt.figure(num='map position {} - after train'.format(pos))
plt.imshow(np.reshape(som.map[pos], (8, 8)), cmap='bone')
plt.show()