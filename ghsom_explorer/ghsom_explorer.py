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
