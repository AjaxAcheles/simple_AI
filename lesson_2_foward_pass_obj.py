import numpy as np

np.random.seed(0)

################### INPUTS ###################
# each row (inputs[x]) is a snapshot of data from different points in time.
# each element (inputs[x][y]) is the data from an input neurons.
X = [
            [1, 2, 3, 2.5], # data at time 1
            [2.0, 5.0, -1.0, 2.0], # data at time 2
            [-1.5, 2.7, 3.3, -0.8] # data at time 3
]
#############################################

class LayerDense:
    # n_inputs is the number of inputs to the layer
    # n_neurons is the number of neurons in the layer
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # a bias of 0 is assigned to each neuron in the layer
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# initialize foward pass
amount_of_neurons_in_each_layer = [4, 12, 6, 1]
layer1 = LayerDense(amount_of_neurons_in_each_layer[0], amount_of_neurons_in_each_layer[1])
layer2 = LayerDense(amount_of_neurons_in_each_layer[1], amount_of_neurons_in_each_layer[2])
layer3 = LayerDense(amount_of_neurons_in_each_layer[2], amount_of_neurons_in_each_layer[3])
# execute foward pass
layer1.forward(X)
layer2.forward(layer1.output)
layer3.forward(layer2.output)
print("Output Layer Output:", layer3.output)

