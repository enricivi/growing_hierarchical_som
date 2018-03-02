import numpy as np
from scipy.ndimage.filters import gaussian_filter


# TODO: add docstring
class SOM:
    def __init__(self, map_dim, metrics=None, learning_rate=0.5, update_mask_dim=(5, 5), sigma=1.0, decay=0.99):
        """
        :param map_dim: is a 3-dimensional tuple (row, column, number of features per example)
        :param metrics: metrics used to compute the closest neuron (if None metrics will be euclidean distance)
        :param learning_rate: step dimension during the training
        :param update_mask_dim: how much neurons update per example (odd number, odd number)
        :param sigma: standard deviation of the gaussian that define the update mask
        :param decay: learning rate decay factor
        """
        self.map = np.empty(shape=map_dim, dtype=np.float32)
        self.map_dim = { 'w':  map_dim[0], 'h':  map_dim[1], 'depth':  map_dim[2] }

        self.learning_rate = learning_rate
        self.decay = decay

        self.update_mask = np.zeros(shape=update_mask_dim, dtype=np.float32)
        self.update_mask[int((update_mask_dim[0]-1)/2), int((update_mask_dim[1]-1)/2)] = 1.0
        self.update_mask = gaussian_filter(self.update_mask, sigma=sigma)

        self.metrics = metrics
        if metrics is None:
            self.metrics = lambda x, y: np.linalg.norm(x-y)

    def rand_init_from_data(self, data, seed=None):
        """ initializes the map of the SOM picking random samples from data """
        random_generator = np.random.RandomState(seed)
        for idx in np.ndindex(self.map_dim["w"], self.map_dim["h"]):
            self.map[idx] = data[int(random_generator.rand()*(len(data) - 1))]
            self.map[idx] /= np.linalg.norm(self.map[idx])

    def activate(self, X):
        """
        :param X: a sample from the data (length = map_dim[2])
        """
        activations = np.empty(shape=(self.map_dim["w"], self.map_dim["h"]), dtype=np.float32)
        it = np.nditer(activations, flags=['multi_index'])
        while not it.finished:
            activations[it.multi_index] = self.metrics(X, self.map[it.multi_index[0], it.multi_index[1]])
            it.iternext()
        return activations

    def train(self, data, epochs=5, seed=None):
        """
        updates the weights of the neurons
        """
        print("\nstart training")
        random_generator = np.random.RandomState(seed)
        lr = self.learning_rate
        for epoch in range(epochs):
            for idx in random_generator.permutation(len(data)):
                lr = lr*self.decay

                x = data[idx]
                winner = self.min_activation_idx(x)

                dx = int((self.update_mask.shape[0]-1)/2)
                dy = int((self.update_mask.shape[1]-1)/2)
                it = np.nditer(self.update_mask, flags=['multi_index'])
                while not it.finished:
                    i = winner[0] + (it.multi_index[0] - dx)
                    j = winner[1] + (it.multi_index[1] - dy)
                    if (i >= 0) and (j >= 0) and (i < self.map.shape[0]) and (j < self.map.shape[1]):
                        self.map[i, j] += lr*self.update_mask[it.multi_index]*(x - self.map[i, j])
                        self.map[i, j] /= np.linalg.norm(self.map[i, j])
                    it.iternext()
            print( "   --> epoch {} (of {}) finished".format(epoch+1, epochs) )
        print("end training\n")

    def min_activation_idx(self, data):
        activations = self.activate(data)
        return np.unravel_index(activations.argmin(), activations.shape)

    def heat_map(self, dataset):
        heat_map = np.zeros(shape=(self.map_dim["w"], self.map_dim["h"]))
        for data in dataset:
            winner = self.min_activation_idx(data)
            heat_map[winner] += 1
        return heat_map
