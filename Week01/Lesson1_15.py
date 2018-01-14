import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    exp_l = np.exp(L)
    return list(map(lambda a: np.exp(a) / exp_l.sum(), L))

print(softmax([5, 6, 7]))