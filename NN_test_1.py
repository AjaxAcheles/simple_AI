import math
from random import uniform

class Weight:
    def __init__(self, value):
        self.value = value


class Bias:
    def __init__(self, value):
        self.value = 0


class Neuron:
    def __init__(self, num_inputs, activation_function, *args, **kwargs):
        # set weights
        if kwargs.get('weights') and len(kwargs.get('weights')) == num_inputs:
            self.weights = [Weight(uniform(-1, 1)) for input in range(num_inputs)]
        # set bias
        if kwargs.get('bias'):
            self.bias = Bias(kwargs.get('bias'))
        else:
            self.bias = Bias(0)




class HiddenLayer:
    def __init__(self, num_inputs, num_neurons, activation_function):
        self.neurons = self.populate_layer(num_inputs, num_neurons, activation_function)
    
    def populate_layer(self, num_inputs, num_neurons, activation_function):
        neurons = [Neuron(num_inputs, activation_function) for neuron_index in range(num_neurons)]
        return neurons


class InputLayer(HiddenLayer):
    def __init__(self, num_neurons):
        super().__init__(None, num_neurons, ActivationFunctions.linear)
    
    def populate_layer(self, num_inputs, num_neurons, activation_function):
        neurons = [Neuron(0, activation_function, bias=0) for neuron_index in range(num_neurons)]
        return neurons
    
    def set_inputs(self, inputs):
        for neruon_index in range(len(self.neurons)):
            neuron.output = inputs[neruon_index]


class OutputLayer(HiddenLayer):
    def __init__(self, num_inputs, num_neurons):
        super().__init__(num_inputs, num_neurons, ActivationFunctions.linear)

    def populate_layer(self, num_inputs, num_neurons, activation_function):
        neurons = [Neuron(num_inputs, activation_function) for neuron_index in range(num_neurons)]
        return neurons


class NeuralNetwork:
    def __init__(self, *layers, **kwargs):
        # layers will be structured like this: [num_neurons, activation_function]
        self.layers = []
        for layer_index in range(len(layers)):
            if layer_index == 0:
                # if input layer
                current_layer_size = layers[layer_index][0]
                layer = InputLayer(current_layer_size)
            elif layer_index == len(layers) - 1:
                # if output layer
                last_layer_size = layers[layer_index - 1][0]
                current_layer_size = layers[layer_index][0]
                layer = OutputLayer(last_layer_size, current_layer_size)
            else:
                # if hidden layer
                last_layer_size = layers[layer_index - 1][0]
                current_layer_size = layers[layer_index][0]
                activation_function = layers[layer_index][1]
                layer = HiddenLayer(last_layer_size, current_layer_size, activation_function)
                        
            self.layers.append(layer)
        print([layer.neurons for layer in self.layers])

        # set other variables
        if kwargs.get("cost_function"):
            self.cost_function = kwargs.get("cost_function")
        else:
            self.cost_function = ErrorFunctions.CostFunctions.mean_squared_error

        if kwargs.get("learning_rate"):
            self.learning_rate = kwargs.get("learning_rate")
        else:
            self.learning_rate = 0.1





class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return sigmoid(x) * (1 - sigmoid(x))

    @staticmethod
    def relu(x):
        return max(0, x)

    @staticmethod
    def relu_derivative(x):
        if x > 0:
            return 1
        else:
            return 0
    
    @staticmethod
    def tanh(x):
        return math.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - math.tanh(x)
    
    @staticmethod
    def linear(x):
        return x

class ErrorFunctions:
    # this site explains the diffenence between cost and loss functions
    # https://www.baeldung.com/cs/cost-vs-loss-vs-objective-function
    class CostFunctions:
        @staticmethod
        def mean_squared_error(y_pred, y_true):
            if len(y_pred) == len(y_true):
                sum([ErrorFunctions.LossFunctions.squared_error(y_pred[y_index], y_true[y_index]) for y_index in range(len(y_true))])
            else:
                raise ValueError("y_pred and y_true must be the same length")

    class LossFunctions:
        @staticmethod
        def squared_error(y_pred, y_true):
            return (y_pred - y_true) ** 2


# create neural network
NN = NeuralNetwork([1], [2, ActivationFunctions.tanh], [2])

# forward pass

# backpropagation
# take derivative of loss function for all weights and biases. ie: gradient of loss funciton
# plug each perameter value into the derivatives. exmpl: dc_dw1(w1)
# calc step size using learning rate
# update params