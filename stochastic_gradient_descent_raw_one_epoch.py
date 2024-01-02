import math
from random import randint

# data set

def make_data_set(number_of_points):
    X = [randint(0, 1) for index in range(number_of_points)]
    y = [(-1 * (data_point - 1)) for data_point in X]
    return X, y

X, y = make_data_set(10000)


# retrive weights and biases from file and re-assign values to neural network variables
with open("stochastic_gradient_descent_raw_one_epoch_NN_data.txt", "r") as file:
    lines = file.readlines()
    w1 = float(lines[0].split("=")[1])
    b1 = float(lines[1].split("=")[1])
    w2 = float(lines[2].split("=")[1])
    b2 = float(lines[3].split("=")[1])
    w3 = float(lines[4].split("=")[1])
    b3 = float(lines[5].split("=")[1])

def sigmoid(result):
    # sigmoid activation_function
    return 1 / (1 + math.pow(math.e, -result))

# training
for index in range(len(X) - 1):
    data_point = X[index]
    y_true = y[index]
    # forward pass
    z1 = data_point * w1 + b1
    r1 = sigmoid(z1)
    z2 = r1 * w2 + b2
    r2 = sigmoid(z2)
    r3 = w3 * r2 + b3
    
    # backward pass
    c = math.pow(r3 - y_true, 2)
    learning_rate = 0.1
    
    # output layer
    dc_dr3 = 2 * (r3 - y_true)
    dr3_dw3 = r2
    dr3_db3 = 1
    
    dc_dw3 = dc_dr3 * dr3_dw3
    dc_db3 = dc_dr3 * dr3_db3
    
    
    # hidden layer 2
    dc_dr3 = 2 * (r3 - y_true)
    dr3_dr2 = w3
    dr2_dz2 = (1 / (1 + math.pow(math.e, -z2))) * (1 - (1 / (1 + math.pow(math.e, -z2))))
    dz2_dw2 = r1
    dz2_db2 = 1
    
    dc_dw2 = dc_dr3 * dr3_dr2 * dr2_dz2 * dz2_dw2
    dc_db2 = dc_dr3 * dr3_dr2 * dr2_dz2 * dz2_db2
    
    
    # hidden layer 2
    dc_dr3 = 2 * (r3 - y_true)
    dr3_dr2 = w3
    dr2_dz2 = (1 / (1 + math.pow(math.e, -z2))) * (1 - (1 / (1 + math.pow(math.e, -z2))))
    dz2_dr1 = w2
    dr1_dz1 = (1 / (1 + math.pow(math.e, -z1))) * (1 - (1 / (1 + math.pow(math.e, -z1))))
    dz1_dw1 = data_point
    dz1_db1 = 1
    
    dc_dw1 = dc_dr3 * dr3_dr2 * dr2_dz2 * dz2_dr1 * dr1_dz1 * dz1_dw1
    dc_db1 = dc_dr3 * dr3_dr2 * dr2_dz2 * dz2_dr1 * dr1_dz1 * dz1_db1
    
    
    # update weights
    w1 = w1 - learning_rate * dc_dw1
    b1 = b1 - learning_rate * dc_db1
    w2 = w2 - learning_rate * dc_dw2
    b2 = b2 - learning_rate * dc_db2
    w3 = w3 - learning_rate * dc_dw3
    b3 = b3 - learning_rate * dc_db3
    
# final test forward pass
    z1 = X[-1] * w1 + b1
    r1 = sigmoid(z1)
    z2 = r1 * w2 + b2
    r2 = sigmoid(z2)
    r3 = w3 * r2 + b3
print(f"{X[-1]}--{w1}-->+{b1}+{z1}-o-{r1}--{w2}-->+{b2}+{z2}-o-{r2}--{w3}-->+{b3}+{r3}->'FINAL y-pred'| y-true = {y[-1]}")

# store weights and biases in "stochastic_gradient_descent_raw_one_epoch_NN_data.txt" file
with open("stochastic_gradient_descent_raw_one_epoch_NN_data.txt", "w") as file:
    file.write(f"w1 = {w1}\nb1 = {b1}\nw2 = {w2}\nb2 = {b2}\nw3 = {w3}\nb3 = {b3}")
