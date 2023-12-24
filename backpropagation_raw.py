import numpy as np

################### INPUTS ###################
# each row (inputs[x]) is a snapshot of data from different points in time.
# each element (inputs[x][y]) is the data from an input neurons.
X = [
            [1.0], # data at time 1
            [0.0], # data at time 2
            [0.0], # data at time 3
            [0.0], # data at time 4
            [1.0], # data at time 5
            [0.0], # data at time 6
            [0.0], # data at time 7
            [1.0], # data at time 8
            [1.0], # data at time 9
            [1.0], # data at time 10
            [0.0], # data at time 11
            [0.0], # data at time 12
            [1.0], # data at time 13
            [1.0], # data at time 14
            [0.0], # data at time 15
            [0.0], # data at time 16
            [0.0], # data at time 17
            [1.0], # data at time 18
            [1.0], # data at time 19
            [0.0] # data at time 20
]
X =  np.array(X)

y = [
    [0.0], # data at time 1
    [1.0], # data at time 2
    [1.0], # data at time 3
    [1.0], # data at time 4
    [0.0], # data at time 5
    [1.0], # data at time 6
    [1.0], # data at time 7
    [0.0], # data at time 8
    [0.0], # data at time 9
    [0.0], # data at time 10
    [1.0], # data at time 11
    [1.0], # data at time 12
    [0.0], # data at time 13
    [0.0], # data at time 14
    [1.0], # data at time 15
    [1.0], # data at time 16
    [1.0], # data at time 17
    [0.0], # data at time 18
    [0.0], # data at time 19
    [1.0] # data at time 20
]
y = np.array(y)
#############################################
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

################### LAYER ONE ###################
# each row is a neuron in which each element is a different neuron's weights for each input.
weights1 = [
            [0.0], # weight from neuron 0 (input neuron) to neuron 1 in layer 1
]
weights1 = np.array(weights1)
# each element is the bias of an individual neuron 
bias1 = [
        0.0, # neuron 1 in layer 1
]
bias1 = np.array(bias1)
# transpose weights matrix so that the network can eval the proper time snapshots of data.
# aka: so that the dot product works.
output1 = np.dot(X, np.array(weights1).T) + bias1
output1 = sigmoid_function(output1)
print("Layer 1 Output:", output1)
###### END OF LAYER ONE ###################

################### OUTPUT LAYER ###################

inputs2 = output1

weights2 = [
            [0.11], # weight from neuron 1 in layer 1 to neuron 2 in layer 2 (output neuron)

]
weights2 = np.array(weights2)

bias2 = [
        0.22, # neuron 2
]
bias2 = np.array(bias2)

# ReLU
output2 = np.dot(inputs2, np.array(weights2).T) + bias2
print("Layer 2 Output:", output2)
################### END OF OUTPUT LAYER ###################

################### Backpropagation ###################

# Assume you have a loss function that computes the mean squared error between the output and the target
loss = np.mean((output2 - y) ** 2)

# Compute the gradients of the loss with respect to the weights and biases of each layer using the chain rule
# For simplicity, I will use the variable names dL_dw1, dL_db1, dL_dw2, dL_db2 to denote these gradients

# Layer 2 gradients
dL_dout2 = 2 * (output2 - y) # derivative of MSE loss with respect to output2
dout2_din2 = np.where(inputs2 > 0, 1, 0) # derivative of ReLU activation with respect to inputs2
dL_din2 = dL_dout2 * dout2_din2 # chain rule
dL_dw2 = np.dot(dL_din2.T, output1) # derivative of inputs2 with respect to weights2
dL_db2 = np.sum(dL_din2, axis=0) # derivative of inputs2 with respect to bias2

# Layer 1 gradients
dL_dout1 = np.dot(dL_din2, weights2.T) # derivative of loss with respect to output1
dout1_din1 = output1 * (1 - output1) # derivative of sigmoid activation with respect to output1
dL_din1 = dL_dout1 * dout1_din1 # chain rule
dL_dw1 = np.dot(dL_din1.T, X) # derivative of output1 with respect to weights1
dL_db1 = np.sum(dL_din1, axis=0) # derivative of output1 with respect to bias1

# Update the weights and biases of each layer using a learning rate
lr = 0.01 # you can choose a different value for this
weights2 -= lr * dL_dw2 # gradient descent update
bias2 -= lr * dL_db2
weights1 -= lr * dL_dw1
bias1 -= lr * dL_db1

