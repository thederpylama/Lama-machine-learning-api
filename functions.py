import numpy as np
# import pandas as pd
# import math
# import mpmath as mp


# function for one hot encoding
def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


# chooses the greatest value form the list of probabilities
def get_predictions(a):
    return np.argmax(a, 0)


# checks the accuracy of the model's predictions
def get_accuracy(predictions, labels):
    return np.sum(predictions == labels) / labels.size
