import numpy as np

################### INPUTS ###################
# each row (inputs[x]) is a snapshot of data from different points in time.
# each element (inputs[x][y]) is the data from an input neurons.
inputs1 = [
            [1, 2, 3, 2.5], # data at time 1
            [2.0, 5.0, -1.0, 2.0], # data at time 2
            [-1.5, 2.7, 3.3, -0.8] # data at time 3
]
#############################################

################### LAYER ONE ###################
# each row is a neuron in which each element is a different neuron's weights for each input.
weights1 = [
            [0.2, 0.8, -0.5, 1.8], # neuron 1
            [0.5, -0.91, 0.26, 2.3], # neuron 2
            [-0.26, -0.27, 0.17, 0.87] # neuron 3
]
# each element is the bias of an individual neuron 
bias1 = [
        2, # neuron 1
        3, # neuron 2
        0.5 # neuron 3
]
# transpose weights matrix so that the network can eval the proper time snapshots of data.
# aka: so that the dot product works.
output1 = np.dot(inputs1, np.array(weights1).T) + bias1
print("Layer 1 Output:", output1)
# example output:
# each row contains the neural network's results for each given time snapshot of data
# output = [
#           [ 6.8    8.21   2.385], # each element is the output of each neuron
#           [ # this is one row spread out for visulization purposes
#           10.5, # output of neuron 1
#           3.79, # output of neuron 2
#           0.2 # output of neuron 3
#           ],
#           [ 0.77  -1.189  0.026]
# ]
################### END OF LAYER ONE ###################

################### LAYER TWO ###################

inputs2 = output1

weights2 = [
            [0.1, -0.14, 0.5], # neuron 1
            [-0.5, 0.12, -0.33], # neuron 2
            [-0.44, 0.73, -0.13] # neuron 3
]

bias2 = [
        -1, # neuron 1
        2, # neuron 2
        -0.5 # neuron 3
]

output2 = np.dot(inputs2, np.array(weights2).T) + bias2
print("Layer 2 Output:", output2)
################### END OF LAYER TWO ###################

################### OUTPUT LAYER ###################

inputs3 = output2

# create a layer with one neuron
weights3 = [
            [
                0.1, # weight for connection between layer 2 neuron 1 and output layer neuron.
                0.2, # weight for connection between layer 2 neuron 2 and output layer neuron.
                0.3  # weight for connection between layer 2 neuron 3 and output layer neuron.
            ]
]

bias3 = [
            -0.1 # bias output layer neuron
]

output3 = np.dot(inputs3, np.array(weights3).T) + bias3
print("Output Layer Output:", output3)
################### END OF OUTPUT LAYER ###################