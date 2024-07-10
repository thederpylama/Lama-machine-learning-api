import numpy as np
# import pandas as pd
# import math
# import mpmath as mp


# ReLU activation function
def relu(z):
    return np.maximum(z, 0)


# derivative of ReLU for back propagation
def deriv_relu(z):
    return z > 0


# normalizes the values of a vector so that they can be interpreted as probabilities
def softmax(z):
    return np.exp(z) / sum(np.exp(z))
