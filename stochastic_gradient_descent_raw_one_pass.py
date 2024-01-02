import math

X = 1
y = 0


# hidden layer 1
w1 = 0.01
b1 = 0.0
def hidden_layer_1_activation_function(result):
    # sigmoid activation_function
    return 1 / (1 + math.pow(math.e, -result))

# hidden layer 2
w2 = -0.01
b2 = 0.0
def hidden_layer_2_activation_function(result):
    # sigmoid activation_function
    return 1 / (1 + math.pow(math.e, -result))

# output layer
w3 = 0.01
b3 = 0.0

# forward pass
z1 = X * w1 + b1
r1 = hidden_layer_1_activation_function(z1)
z2 = r1 * w2 + b2
r2 = hidden_layer_2_activation_function(z2)
r3 = w3 * r2 + b3
print(r3)

# backward pass
c = (r3 - y) ** 2
learning_rate = 0.01

# output layer
dc_dr3 = 2 * (r3 - y)
dr3_dw3 = r2
dr3_db3 = 1

dc_dw3 = dc_dr3 * dr3_dw3
dc_db3 = dc_dr3 * dr3_db3


# hidden layer 2
dc_dr3 = 2 * (r3 - y)
dr3_dr2 = w3
dr2_dz2 = (1 / (1 + math.pow(math.e, -z2))) * (1 - (1 / (1 + math.pow(math.e, -z2))))
dz2_dw2 = r1
dz2_db2 = 1

dc_dw2 = dc_dr3 * dr3_dr2 * dr2_dz2 * dz2_dw2
dc_db2 = dc_dr3 * dr3_dr2 * dr2_dz2 * dz2_db2


# hidden layer 2
dc_dr3 = 2 * (r3 - y)
dr3_dr2 = w3
dr2_dz2 = (1 / (1 + math.pow(math.e, -z2))) * (1 - (1 / (1 + math.pow(math.e, -z2))))
dz2_dr1 = w2
dr1_dz1 = (1 / (1 + math.pow(math.e, -z1))) * (1 - (1 / (1 + math.pow(math.e, -z1))))
dz1_dw1 = X
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
print(f"{X}--{w1}-->+{b1}+{z1}-o-{r1}--{w2}-->+{b2}+{z2}-o-{r2}--{w3}-->+{b3}+{r3}->'y-pred'")

