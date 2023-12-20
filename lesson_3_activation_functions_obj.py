import numpy as np
np.random.seed(0)
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class LayerSimple:
    # n_inputs is the number of inputs to the layer
    # n_neurons is the number of neurons in the layer
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # a bias of 0 is assigned to each neuron in the layer
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        neural_network_output = np.dot(inputs, self.weights) + self.biases
        self.output = neural_network_output

class LayerReLU:
    # n_inputs is the number of inputs to the layer
    # n_neurons is the number of neurons in the layer
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # a bias of 0 is assigned to each neuron in the layer
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        neural_network_output = np.dot(inputs, self.weights) + self.biases
        self.output = self.rectified_linear_unit(neural_network_output)


    def rectified_linear_unit(self, outputs):
        # rectified linear unit activation function (ReLU)
        return np.maximum(0, outputs)

class LayerSoftmax:
    # n_inputs is the number of inputs to the layer
    # n_neurons is the number of neurons in the layer
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # a bias of 0 is assigned to each neuron in the layer
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        neural_network_output = np.dot(inputs, self.weights) + self.biases
        self.output = self.softmax(neural_network_output)

    def softmax(self, outputs):
        exp_func_values = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
        normalized_values = exp_func_values / np.sum(exp_func_values, axis=1, keepdims=True)
        return normalized_values


################### INPUTS ###################
# each row (inputs[x]) is a snapshot of data from different points in time.
# each element (inputs[x][y]) is the data from an input neurons.
#X = [
#    [4.8, 1.21, 2.385],
#    [8.9, -1.81, 0.2],
#    [1.41, 1.33, 0.66]
#    ]
X, y = spiral_data(samples=100, classes=3)
#############################################


layer1_ReLU = LayerReLU(2, 3)
layer2_softmax = LayerSoftmax(3, 3)

layer1_ReLU.forward(X)
layer2_softmax.forward(layer1_ReLU.output)
print(layer2_softmax.output)























