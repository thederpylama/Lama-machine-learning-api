# Lama-machine-learning-api
A TensoreFlow-Keras like machine learning API that uses Numpy for matrix math.
This is a Python machine learning API that is designed to have similar syntax to TensorFlow-Keras.
It is written in bare Python, except for the matrix math that is done using Numpy.

Layers Types:

Dense- A fully connected dense layer

Dropout- A layer that randomly sets values in the input matrix to zero at a user-defined rate. This layer is used to prevent overfitting.

Activations:

ReLU- Rectified Linear Unit activation

Softmax- Not exactly an activation function, but is used to normalize the outputs of the last layer so that probabilities can be found

Loss Functions:

Sparse categorical cross entropy- Used for multiclass classification
