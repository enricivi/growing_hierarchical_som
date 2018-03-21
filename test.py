import numpy as np
from sklearn.datasets import load_digits
from time import time
from GHSOM import GHSOM

from ghsom_explorer import explore
from ghsom_explorer import plot


def normalize_dataset(dataset):
    for idx, data in enumerate(dataset):
        norm_data = data / np.linalg.norm(data)
        dataset[idx] = norm_data


digits = load_digits()

data = digits.data
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

print("dataset length: {}".format(n_samples))
print("features per example: {}".format(n_features))
print("number of digits: {}\n".format(n_digits))

ghsom = GHSOM(input_dataset=data, t1=0.2, t2=0.1, learning_rate=0.5, decay=0.7, gaussian_sigma=1, epoch_number=15)
#print("Normalizing dataset...")
#normalize_dataset(data)

print("Training...")
start = time()
zu = ghsom()
print("Elapsed time is {:.2f} seconds\n".format(time() - start))

map_per_level = dict()
explore(zu, map_per_level)
plot(map_per_level, level=1)
plot(map_per_level, level=2)
for n in map_per_level[2]:
    print(n[1].position)
