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
#############################################
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

################### LAYER ONE ###################
# each row is a neuron in which each element is a different neuron's weights for each input.
weights1 = [
            [-1.0], # weight from neuron 0 (input neuron) to neuron 1 in layer 1
]
# each element is the bias of an individual neuron 
bias1 = [
        0, # neuron 1 in layer 1
]
# transpose weights matrix so that the network can eval the proper time snapshots of data.
# aka: so that the dot product works.
output1 = np.dot(X, np.array(weights1).T) + bias1
output1 = sigmoid_function(output1)
print("Layer 1 Output:", output1)
###### END OF LAYER ONE ###################

################### OUTPUT LAYER ###################

inputs2 = output1

weights2 = [
            [4], # weight from neuron 1 in layer 1 to neuron 2 in layer 2 (output neuron)

]

bias2 = [
        0, # neuron 2
]

# ReLU
output2 = np.dot(inputs2, np.array(weights2).T) + bias2
print("Layer 2 Output:", output2)
################### END OF OUTPUT LAYER ###################

################### Backpropagation for weights ###################

dcda = 2.0*(output2 - y)
dadz = (sigmoid_function(output2) - 1.0) / sigmoid_function(output2)
dzdw = output1

dcdw = dzdw*dadz*dcda

nudge =  np.sum(dcdw) / len(dcdw)
print("Weights Nudge:", nudge)

################### Backpropagation for biases ###################

dcda = 2.0*(output2 - y)
dadz = (sigmoid_function(output2) - 1.0) / sigmoid_function(output2)
dzdb = 1

dcdb = dzdb*dadz*dcda

nudge =  np.sum(dcdw) / len(dcdw)
print("Biases Nudge:", nudge)