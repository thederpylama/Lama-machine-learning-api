import numpy as np
# import pandas as pd
# import math
# import mpmath as mp
import functions


# class for sequential models, only supports dense layer atm
class Sequential:
    def __init__(self, modelLayers):
        self.modelLayers = modelLayers
        self.dropLayers = []
        # self.dropCount = 0

    # initializes the weights and biases for all layers except for layer 0
    def compile(self):
        # self.modelLayers[0].comp(None)
        # dropOutCount = 0
        aux = []

        for i in range(len(self.modelLayers)):
            if self.modelLayers[i].__str__() == 'Dropout':
                self.modelLayers[i - 1].set_dropFlag(True)

        for j in range(len(self.modelLayers)):
            if self.modelLayers[j].__str__() == 'Dropout':
                self.dropLayers.append(self.modelLayers[j])
            if self.modelLayers[j].__str__() == 'Dense':
                aux.append(self.modelLayers[j])
        self.modelLayers = aux

        for i in range(len(self.modelLayers) - 1):

            '''
            Attempt at recursive implementation
            Broken needs work
            if self.modelLayers[i + 1].__str__() == 'Dropout':
                print("h")
                self.modelLayers[i].set_dropFlag(True)
                self.dropLayers.append(self.modelLayers.pop(i + 1))
                self.compile()
            '''
            if self.modelLayers[i + 1].__str__() == 'Dense':
                self.modelLayers[i + 1].comp(self.modelLayers[i].get_layer_dim())

        return

    # forward propagation
    def forward_prop(self, train):
        dropCount = 0
        # initialize layer 0 weights and biases
        if self.modelLayers[0].get_prev_layer_dim() is None:
            self.modelLayers[0].set_prev_layer_dim(train.shape[0])
            self.modelLayers[0].comp(None)

        out = []
        res = self.modelLayers[0].forward_pass(train)
        if self.modelLayers[0].dropFlag:
            res[-1] = self.dropLayers[dropCount].drop(res[-1])
            dropCount += 1

        out.append(res)
        for i in range(len(self.modelLayers) - 1):
            res = self.modelLayers[i + 1].forward_pass(res[1])
            if self.modelLayers[i + 1].dropFlag:
                res[-1] = self.dropLayers[dropCount].drop(res[-1])
                dropCount += 1
            out.append(res)

        return out

    # back propagation
    def backward_prop(self, forwardOut, train, labels):
        forwardOut = [train] + forwardOut
        # forwardOut.append(labels)
        out = []
        m = labels.size
        oneHotY = functions.one_hot(labels)

        dZ = forwardOut[-1][-1] - oneHotY
        dW = (1 / m) * dZ.dot(forwardOut[-2][-1].T)
        dB = (1 / m) * np.sum(dZ)

        self.modelLayers[-1].set_dW(dW)
        self.modelLayers[-1].set_dB(dB)

        out.append(dW)
        out.append(dB)

        for i in range(len(self.modelLayers) - 1):

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

        return

    # update the weights and biases using calcs from back_prop
    def update_params(self, alpha):
        for i in range(len(self.modelLayers)):
            if self.modelLayers[i].__str__() == 'Dense':
                self.modelLayers[i].update(alpha)

    # main training loop, also displays accuracy throughout training
    def train(self, train, labels, epochs, learningRate):
        for i in range(epochs):
            forwardOut = self.forward_prop(train)
            self.backward_prop(forwardOut, train, labels)
            self.update_params(learningRate)
            if i % 10 == 0:
                print('Epoch ' + str(i) + '/' + str(epochs))
                print("Accuracy: " + str(functions.get_accuracy(functions.get_predictions(forwardOut[-1][-1]), labels)))

    # evaluates the model's accuracy on the test data to determine how well it generalized
    def eval(self, test, testLabels):
        # check accuracy on test data
        for i in range(len(self.modelLayers)):
            if self.modelLayers[i].__str__() == 'Dense':
                self.modelLayers[i].set_dropFlag(False)
        forwardOut = self.forward_prop(test)
        acc = functions.get_accuracy(functions.get_predictions(forwardOut[-1][-1]), testLabels)
        print("Accuracy on test data: " + str(acc))

    # allows the model to be used to make predictions on new data
    def predict(self, new):
        # allow for user to run new data through the model
        forwardOut = self.forward_prop(new)
        prediction = functions.get_predictions(forwardOut[-1][-1])
        print("Prediction: " + str(prediction))
