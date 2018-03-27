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


def plot(gmap):
    plt.figure(num="parent")
    plt.imshow(__gmap_to_matrix(gmap.weights_map), cmap='bone_r', interpolation='sinc')
    for neuron in gmap.neurons.values():
        if neuron.child_map is not None:
            plt.figure(num=str(neuron.position))
            plt.imshow(__gmap_to_matrix(neuron.child_map.weights_map), cmap='bone_r', interpolation='sinc')
    plt.show()


digits = load_digits()

data = digits.data
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

print("dataset length: {}".format(n_samples))
print("features per example: {}".format(n_features))
print("number of digits: {}\n".format(n_digits))

ghsom = GHSOM(input_dataset=data, t1=0.15, t2=0.1, learning_rate=0.5, decay=0.7, gaussian_sigma=1,
              epochs_number=15, dataset_percentage=0.25)

print("Training...")
start = time()
zu = ghsom.train(seed=1)
print("Elapsed time is {:.2f} seconds\n".format(time() - start))

print(zu)
plot(zu.child_map)
