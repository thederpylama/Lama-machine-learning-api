import numpy as np
# import pandas as pd
# import math
# import mpmath as mp
import functions


# class for sequential models, only supports dense layer atm
class Sequential:
    def __init__(self, modelLayers):
        self.modelLayers = modelLayers

    # initializes the weights and biases for all layers except for layer 0
    def compile(self):
        # self.modelLayers[0].comp(None)
        for i in range(len(self.modelLayers) - 1):
            self.modelLayers[i + 1].comp(self.modelLayers[i].get_layer_dim())
        return

    # forward propagation
    def forward_prop(self, train):
        print(self.modelLayers[0].get_prev_layer_dim())

        # initialize layer 0 weights and biases
        if self.modelLayers[0].get_prev_layer_dim() is None:
            print(train.shape[0])
            self.modelLayers[0].set_prev_layer_dim(train.shape[0])
            self.modelLayers[0].comp(None)

        out = []
        res = self.modelLayers[0].forward_pass(train)
        out.append(res)
        for i in range(len(self.modelLayers) - 1):
            print("i")
            print(i)
            res = self.modelLayers[i + 1].forward_pass(res[1])
            out.append(res)

        '''
        print('out')
        copy = np.array(out)
        print(copy.shape)
        '''

        return out

    # back propagation
    def backward_prop(self, forwardOut, train, labels):
        forwardOut = [train] + forwardOut
        # forwardOut.append(labels)
        out = []
        m = labels.size
        oneHotY = functions.one_hot(labels)

        # print("hot")
        # print(forwardOut[-1][-1].shape)
        dZ = forwardOut[-1][-1] - oneHotY
        dW = (1 / m) * dZ.dot(forwardOut[-2][-1].T)
        # print("dW shape")
        # print(dW.shape)
        dB = (1 / m) * np.sum(dZ)

        self.modelLayers[-1].set_dW(dW)
        self.modelLayers[-1].set_dB(dB)

        out.append(dW)
        out.append(dB)

        for i in range(len(self.modelLayers) - 1):
            # print("forwardOut")
            # print(forwardOut)
            # print("Layer ", i + 2)
            # print(forwardOut[-3 - i][-1])
            # print(forwardOut[-3 - i])

            # handle the case of the last layer
            if i == len(self.modelLayers) - 2:
                res = self.modelLayers[-2 - i].backward_pass(forwardOut[-2 - i][-2],
                                                             forwardOut[-3 - i],
                                                             dZ,
                                                             self.modelLayers[-1 - i].get_weights(),
                                                             m)
            # case for layers between first and last
            else:
                res = self.modelLayers[-2 - i].backward_pass(forwardOut[-2 - i][-2],
                                                             forwardOut[-3 - i][-1],
                                                             dZ,
                                                             self.modelLayers[-1 - i].get_weights(),
                                                             m)
            dZ = res

            # out.append(res[1])
            # out.append(res[2])

        return

    # update the weights and biases using calcs from back_prop
    def update_params(self, alpha):
        for i in range(len(self.modelLayers)):
            print("params layer ", i + 1)
            if self.modelLayers[i].__str__() == 'Dense':
                self.modelLayers[i].update(alpha)

    # main training loop, also displays accuracy throughout training
    def train(self, train, labels, epochs, learningRate):
        for i in range(epochs):
            print("Epoch ", i + 1)
            forwardOut = self.forward_prop(train)
            self.backward_prop(forwardOut, train, labels)
            self.update_params(learningRate)
            if i % 10 == 0:
                print('Epoch ' + str(i) + '/' + str(epochs))
                print("Accuracy: " + str(functions.get_accuracy(functions.get_predictions(forwardOut[-1][-1]), labels)))

    # evaluates the model's accuracy on the test data to determine how well it generalized
    def eval(self, test, testLabels):
        # check accuracy on test data
        forwardOut = self.forward_prop(test)
        acc = functions.get_accuracy(functions.get_predictions(forwardOut[-1][-1]), testLabels)
        print("Accuracy on test data: " + str(acc))

    # allows the model to be used to make predictions on new data
    def predict(self, new):
        # allow for user to run new data through the model
        forwardOut = self.forward_prop(new)
        prediction = functions.get_predictions(forwardOut[-1][-1])
        print("Prediction: " + str(prediction))
