import numpy as np
np.random.seed(0)
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer:
    # n_inputs is the number of inputs to the layer
    # n_neurons is the number of neurons in the layer
    def __init__(self, n_inputs, n_neurons):
        self.weights = (0.1 * np.random.randn(n_neurons, n_inputs))
        # a bias of 0 is assigned to each neuron in the layer
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        neural_network_output = np.dot(inputs, self.weights.T) + self.biases
        self.output = self.activation_function(neural_network_output)

    def activation_function(self, output):
        error = "Activation function not implemented"
        raise NotImplementedError(error)

class LayerLinear(Layer):
    def __init__(self, n_inputs, n_neurons):
        self.activation_function_name = "Linear"
        super().__init__(n_inputs, n_neurons)

    def activation_function(self, output):
        # linear activation function
        return output

class LayerReLU(Layer):
    def __init__(self, n_inputs, n_neurons):
        self.activation_function_name = "ReLU"
        super().__init__(n_inputs, n_neurons)

    def activation_function(self, output):
        # rectified linear unit activation function (ReLU)
        return np.maximum(0, output)

class LayerSoftmax(Layer):
    def __init__(self, n_inputs, n_neurons):
        self.activation_function_name = "Softmax"
        super().__init__(n_inputs, n_neurons)

    def activation_function(self, output):
        # Softmax activation function
        # This function will always output values between 0 and 1
        exp_func_values = np.exp(output - np.max(output, axis=1, keepdims=True))
        normalized_values = exp_func_values / np.sum(exp_func_values, axis=1, keepdims=True)
        return normalized_values

class LayerSigmoid(Layer):
    def __init__(self, n_inputs, n_neurons):
        self.activation_function_name = "Sigmoid"
        super().__init__(n_inputs, n_neurons)

    def activation_function(self, output):
        # sigmoid activation function
        return 1 / (1 + np.exp(-output))


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

class LossMSE(Loss):
    # MSE stands for Mean Squared Error
    def loss_function(self, y_prediction, y_true):
        one_hot_y_true = convert_to_one_hot(y_true)

        try:
            loss = np.mean((y_prediction - one_hot_y_true) ** 2, axis=1)
        except:
            error = "y classes and last layer output neurons must be the same number."
            raise error
        return loss
    
############ Backpropagation #################

class Backpropagation:
    def __init__(self, layers, y_prediction, y_true, loss):
        self.layers = layers
        self.y_prediction = y_prediction
        self.y_true = y_true
        self.one_hot_y_true = convert_to_one_hot(y_true)
        self.loss = loss


    def backpropagate(self):
        # Compute the gradients of the loss with respect to the weights and biases of each layer using the chain rule        
        # must compute adjustments from last layer to first layer, hece the `reversed(self.layers)`
        max_layer_index = len(self.layers) - 1
        current_layer_index = len(self.layers) - 1
        saved_dL_dout = [0 for placeholder in range(max_layer_index + 1)]
        # loop through layers in reverse order
        for layer in reversed(self.layers):
            # compute the derivative of the loss with respect to the output of the current layer (dL_dout)

            # if computing backpropagation for last layer, dL_dout is the derivative of loss function
            if max_layer_index == current_layer_index:
                ############# change this to adapt to the type of loss function in use ###############
                dL_dout = 2 * (self.y_prediction - self.one_hot_y_true)
            # else, get the dot product of the last known dL_dout and the weights of the last layer
            else:
                dL_dout = np.dot(saved_dL_dout[current_layer_index + 1], 
                                self.layers[current_layer_index + 1].weights.T)
            
            # convert `dL_dout` to numpy array
            dL_dout = np.array(dL_dout)
            # log dL_dout for future use
            saved_dL_dout[current_layer_index] = dL_dout

            # compute the derivative of the activation function with respect to the input of the layer (dout_din)
            if layer.activation_function_name == "Linear":
                dout_din = layer.output

            elif layer.activation_function_name == "ReLU":
                dout_din = np.where(layer.output > 0, layer.output, 0)

            elif layer.activation_function_name == "Sigmoid":
                dout_din = layer.output * (1 - layer.output)
                
            elif layer.activation_function_name == "Softmax":
                dout_din = np.log(layer.output)

            # convert `dout_din` to numpy array
            dout_din = np.array(dout_din)
            print("dout_din", dout_din)
            
            # apply chain rule
            dL_din = dL_dout * dout_din
            
            # compute new weights and biases for current layer
            dL_dw = np.dot(dL_din.T, layer.output)
            dL_db = np.sum(dL_din, axis=0)

            print("dL_dout", dL_dout)
            print("weights", layer.weights)

            # update weights and biases
            layer.weights = dL_dw - layer.weights
            layer.biases = dL_db - layer.biases

            # increment down `current_layer_index` because loop has ended
            current_layer_index -= 1

        ## Layer 2 gradients
        #dL_dout2 = 2 * (self.y_prediction - self.y_true) # derivative of MSE loss with respect to output2
        #dout2_din2 = np.where(inputs2 > 0, 1, 0) # derivative of ReLU activation with respect to inputs2
        #dL_din2 = dL_dout2 * dout2_din2 # chain rule
        #dL_dw2 = np.dot(dL_din2.T, output1) # derivative of inputs2 with respect to weights2
        #dL_db2 = np.sum(dL_din2, axis=0) # derivative of inputs2 with respect to bias2

        ## Layer 1 gradients
        #dL_dout1 = np.dot(dL_din2, weights2.T) # derivative of loss with respect to output1
        #dout1_din1 = output1 * (1 - output1) # derivative of sigmoid activation with respect to output1
        #dL_din1 = dL_dout1 * dout1_din1 # chain rule
        #dL_dw1 = np.dot(dL_din1.T, X) # derivative of output1 with respect to weights1
        #dL_db1 = np.sum(dL_din1, axis=0) # derivative of output1 with respect to bias1

        ## Update the weights and biases of each layer using a learning rate
        #lr = 0.01 # you can choose a different value for this
        #weights2 -= lr * dL_dw2 # gradient descent update
        #bias2 -= lr * dL_db2
        #weights1 -= lr * dL_dw1
        #bias1 -= lr * dL_db1

    def update_weights(self):
        # update weights
        pass
    
    def update_biases(self):
        # update biases
        pass
################## FUNCTIONS ##################

def convert_to_one_hot(y):
    if len(y.shape) == 1:
        # convert to one-hot encoded labels
        one_hot_y = np.array(np.eye(np.max(y) + 1)[y])
    # check for one-hot encoded labels
    elif len(y.shape) == 2:
        one_hot_y = np.array(y)
    return one_hot_y

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
X, y = spiral_data(samples=1, classes=2)
#############################################

# init layers
layer1_ReLU = LayerReLU(2, 4)
layer2_sigmoid = LayerSigmoid(4, 2)
# forward pass
layer1_ReLU.forward(X)
layer2_sigmoid.forward(layer1_ReLU.output)
loss_function = LossMSE()
loss = loss_function.calculate(layer2_sigmoid.output, y)

# init backpropagation
backpropagation = Backpropagation([layer1_ReLU, layer2_sigmoid], layer2_sigmoid.output, y, loss)
# backpropagation
backpropagation.backpropagate()

