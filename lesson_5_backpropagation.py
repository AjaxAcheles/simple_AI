import numpy as np
np.random.seed(0)
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer:
    # n_inputs is the number of inputs to the layer
    # n_neurons is the number of neurons in the layer
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # a bias of 0 is assigned to each neuron in the layer
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        neural_network_output = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation_function(neural_network_output)

    def activation_function(self, outputs):
        error = "Activation function not implemented"
        raise NotImplementedError(error)

class LayerLinear(Layer):
    def activation_function(self, outputs):
        # linear activation function
        return outputs

class LayerReLU(Layer):
    def activation_function(self, outputs):
        # rectified linear unit activation function (ReLU)
        return np.maximum(0, outputs)

class LayerSoftmax(Layer):
    def activation_function(self, outputs):
        # Softmax activation function
        # This function will always output values between 0 and 1
        exp_func_values = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
        normalized_values = exp_func_values / np.sum(exp_func_values, axis=1, keepdims=True)
        return normalized_values

class LayerSigmoid(Layer):
    def activation_function(self, outputs):
        # sigmoid activation function
        return 1 / (1 + np.exp(-outputs))


##################### LOSS ##############################
class Loss:
    def calculate(self, output, target):
        sample_losses = self.loss_function(output, target)
        data_loss = np.mean(sample_losses)
        return data_loss

class LossCategoricalCrossentropy(Loss):
    def loss_function(self, y_prediction, y_true):
        # number of samples in a batch
        samples = len(y_prediction)
        # clip data to prevent division by 0
        # clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_prediction, 1e-7, 1 - 1e-7)
        # check for scaler value labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # check for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # losses
        loss = -np.log(correct_confidences)
        return loss

class LossCostFunction(Loss):
    def loss_function(self, y_prediction, y_true):
        # number of samples in a batch
        samples = len(y_prediction)
        # check for scaler value labels
        if len(y_true.shape) == 1:
            # convert to one-hot encoded labels
            correct_confidences = np.array(np.eye(3)[y_true])
        # check for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.array(y_true)
        # losses
        
        # losses
        # cost = {[(y_prediction[0][0] - y_true[0][0])^2 + (y_prediction[0][1] - y_true[0][1])^2 + (y_prediction[0][2] - y_true[0][2])^2] + [(y_prediction[1][0] - y_true[1][0])^2 + (y_prediction[1][1] - y_true[1][1])^2 + (y_prediction[1]][2] - y_true[1][2])^2] + ...} / len(y_true) / samples
        cost  = np.sum((y_prediction - correct_confidences)**2, axis=1) / samples
        print(f'cost: {cost}')
        return cost
    
############ Backpropagation #################

class Backpropagation:
    def __init__(self, layer_weights, layer_biases, y_prediction, y_true, cost):
        self.layer_weights = layer_weights
        self.layer_biases = layer_biases
        self.y_prediction = y_prediction
        self.y_true = y_true
        self.cost = cost


    def backpropagate(self):
        print(f'y_prediction: {self.y_prediction}')
        print(f'y_true: {self.y_true}')
        # calculate gradient
        pass

    def update_weights(self):
        # update weights
        pass
    
    def update_biases(self):
        # update biases
        pass
    
################### INPUTS ###################
# each row (inputs[x]) is a snapshot of data from different points in time.
# each element (inputs[x][y]) is the data from an input neurons.
#X = [
#    [4.8, 1.21, 2.385],
#    [8.9, -1.81, 0.2],
#    [1.41, 1.33, 0.66]
#    ]

# X is data points on graph
# y is the labels for each data point
# the spiral_data function returns a numpy array of 100 coords each seperated
#into 3 distinct classes
X, y = spiral_data(samples=100, classes=3)
#############################################

# init layers
layer1_ReLU = LayerReLU(2, 15)
layer2_softmax = LayerSoftmax(15, 3)
# forward pass
layer1_ReLU.forward(X)
layer2_softmax.forward(layer1_ReLU.output)
loss_function = LossCategoricalCrossentropy()
cost_function = LossCostFunction()
loss = loss_function.calculate(layer2_softmax.output, y)
cost = cost_function.calculate(layer2_softmax.output, y)

#print(f'output: {layer2_softmax.output}')
print(f'loss: {loss}')
print(f"Cost: {cost}")

# init backpropagation
backpropagation = Backpropagation([layer1_ReLU.weights, layer2_softmax.weights], [layer1_ReLU.biases, layer2_softmax.biases], layer2_softmax.output, y, cost)
# backpropagation
backpropagation.backpropagate()

