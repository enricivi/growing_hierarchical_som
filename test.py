import numpy as np
from sklearn.datasets import load_digits
from time import time
from GHSOM import GHSOM


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

ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.5, learning_rate=0.5, decay=0.7, gaussian_sigma=1, epochs_number=15)
#print("Normalizing dataset...")
#normalize_dataset(data)

print("Training...")
start = time()
zu = ghsom.train(seed=0)
print("Elapsed time is {:.2f} seconds\n".format(time() - start))

print(zu)