import numpy as np
from sklearn.datasets import load_digits
from collections import OrderedDict
from time import time
import matplotlib.pyplot as plt
from GHSOM import GHSOM


digits = load_digits()

data = digits.data
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

print("dataset length: {}".format(n_samples))
print("features per example: {}".format(n_features))
print("number of digits: {}".format(n_digits))

ghsom = GHSOM(data, t1=2, t2=1, learning_rate=0.5, decay=0.99, gaussian_sigma=1, epoch_number=5)

zu = ghsom()