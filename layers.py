import numpy as np
import activations
# import pandas as pd
# import math
# import mpmath as mp


# class for a full connected dense layer
class Dense:
    '''
    def __init__(self, inArray, inDim, outputDim, activation):
        self.inArray = inArray
        self.inDim = inDim
        self.outputSize = outputDim
        self.activation = activation

        self.inArrayShape = self.inArray.shape
        self.weights = np.random.rand(self.inDim, self.inArrayShape[0]) - 0.5
        self.biases = np.random.rand(self.inDim, 1) - 0.5
    '''

    def __init__(self, dim, activation):
        self.dim = dim
        self.weights = None
        self.biases = None
        self.prev_layer_dim = None
        self.dW = None
        self.dB = None
        self.activation = activation

    # override string representation to for identifying layer type
    def __str__(self):
        return "Dense"

    # set weights and biases values
    def comp(self, prev_layer_dim):
        if prev_layer_dim is not None:
            self.prev_layer_dim = prev_layer_dim
            self.weights = np.random.rand(self.dim, self.prev_layer_dim) - 0.5
            self.biases = np.random.rand(self.dim, 1) - 0.5

        # case for first layer
        else:
            print(type(self.prev_layer_dim))
            self.weights = np.random.rand(self.dim, self.prev_layer_dim) - 0.5
            self.biases = np.random.rand(self.dim, 1) - 0.5
            print("h")

    def get_prev_layer_dim(self):
        return self.prev_layer_dim

    def set_prev_layer_dim(self, prev_layer_dim):
        self.prev_layer_dim = prev_layer_dim

    def get_layer_dim(self):
        return int(self.dim)

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases

    def set_dW(self, dW):
        self.dW = dW

    def set_dB(self, dB):
        self.dB = dB

    # preforms the forward pass for a given layer
    def forward_pass(self, inArray):
        z = self.weights.dot(inArray) + self.biases

        # ReLU activation function
        if self.activation == 'relu':
            a = activations.relu(z)
            # print(a)
            # print(a.shape)

        # softmax activation, usually for final layer
        elif self.activation == 'softmax':
            a = activations.softmax(z)

        # add more functions

        # linear activation case
        else:
            a = z
        return z, a

    # performs the backwards pass for a given layer
    def backward_pass(self, z, a, dZ0, w, m):

        if self.activation == 'relu':
            # print("dZ0")
            # print(dZ0.shape)
            print("z")
            print(z.shape)
            dZ1 = w.T.dot(dZ0) * activations.deriv_relu(z)
            print("dZ1")
            print(dZ1.shape)
            print(a.shape)

        # linear activation case, probably needs work
        else:
            dZ1 = w.T.dot(dZ0)
        self.dW = (1 / m) * dZ1.dot(a.T)
        print(self.dW.shape)
        self.dB = (1 / m) * np.sum(dZ1)

        return dZ1

    # update the weights and biases using alpha as the learning rate
    def update(self, alpha):
        print("dW")
        print(self.dW.shape)
        self.weights = self.weights - alpha * self.dW
        self.biases = self.biases - alpha * self.dB
