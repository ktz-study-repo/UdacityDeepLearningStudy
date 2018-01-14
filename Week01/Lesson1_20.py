import numpy as np


# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    np_y, np_p = np.array(Y), np.array(P)
    return (-((np_y * np.log(np_p)) + ((1 - np_y) * np.log(1 - np_p)))).sum()

print(cross_entropy([1, 0, 1, 1], [0.4, 0.6, 0.1, 0.5]))
