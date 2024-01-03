import math
from random import randint

# data set

def make_data_set(number_of_points):
    X = [float(randint(0, 1)) for index in range(number_of_points)]
    print("X generated.")
    y = [(-1.0 * (data_point - 1.0)) for data_point in X]
    return X, y

sample_size = 1000000
X, y = make_data_set(sample_size)

# one alert for every n passes completed
alert_frequency = 10000

# rate at which NN learns
learning_rate = 0.005


w1 = None
b1 = None
w2 = None
b2 = None
w3 = None
b3 = None
# retrive weights and biases from file and re-assign values to neural network variables
with open("stochastic_gradient_descent_raw_small_epoch_NN_data.txt", "r") as file:
    lines = file.readlines()
    w1 = float(lines[0].split("=")[1])
    b1 = float(lines[1].split("=")[1])
    w2 = float(lines[2].split("=")[1])
    b2 = float(lines[3].split("=")[1])
    w3 = float(lines[4].split("=")[1])
    b3 = float(lines[5].split("=")[1])

def ReLu(result):
    if result > 0.0:
        return result
    else:
        return 0

def convert_to_one_hot(y):
    if y == 0:
        return [1, 0]
    elif y == 1:
        return [0, 1]
    else:
        raise Exception("TypeError: Cannot convert to one-hot format")


MA_c_r2 = []
MA_c_r3 = []

# training
for index in range(len(X) - 1):
    data_point = X[index]
    y_true = convert_to_one_hot(y[index])
    # forward pass
    z1 = data_point * w1 + b1
    r1 = ReLu(z1)
    r2 = w2 * r1 + b2
    r3 = w3 * r1 + b3

    # calculate cost
    MA_c_r2.append(math.pow(r2 - y[index], 2))
    MA_c_r3.append(math.pow(r3 - y[index], 2))

    # make sure that there is always 5 data points in MA_c list
    if len(MA_c_r2) > 5:
        MA_c_r2.pop(0)
    if len(MA_c_r3) > 5:
        MA_c_r3.pop(0)
    
    # backwards pass
    
    # output layer 1 (top)
    dc_dr2 = 2 * (r2 - y_true[0])
    dr2_dw2 = r1
    dr2_db2 = 1
    
    dc_dw2 = dc_dr2 * dr2_dw2
    dc_db2 = dc_dr2 * dr2_db2

    # output layer 2 (bottom)
    dc_dr3 = 2 * (r3 - y_true[1])
    dr3_dw3 = r1
    dr3_db3 = 1

    dc_dw3 = dc_dr3 * dr3_dw3
    dc_db3 = dc_dr3 * dr3_db3
    
    # hidden layer 1 (via top)
    dc_dw1 = None

    if y_true == [1, 0]:
        dc_dr2 = 2 * (r2 - y_true[0])
        dr2_dr1 = w2
        dr1_dz1 = 1 if z1 > 0 else 0
        dz1_dw1 = data_point
        dz1_db1 = 1

        dc_dw1 = dc_dr2 * dr2_dr1 * dr1_dz1 * dz1_dw1
        dc_db1 = dc_dr2 * dr2_dr1 * dr1_dz1 * dz1_db1


    # hidden layer 1 (via bottom)
    elif y_true == [0, 1]:
        dc_dr3 = 2 * (r3 - y_true[1])
        dr3_dr1 = w3
        dr1_dz1 = 1 if z1 > 0 else 0
        dz1_dw1 = data_point
        dz1_db1 = 1

        dc_dw1 = dc_dr3 * dr3_dr1 * dr1_dz1 * dz1_dw1
        dc_db1 = dc_dr3 * dr3_dr1 * dr1_dz1 * dz1_db1
    

    # update weights
    w1 = w1 - learning_rate * dc_dw1
    b1 = b1 - learning_rate * dc_db1
    w2 = w2 - learning_rate * dc_dw2
    b2 = b2 - learning_rate * dc_db2
    w3 = w3 - learning_rate * dc_dw3
    b3 = b3 - learning_rate * dc_db3
    if index % alert_frequency == 0:
        print((index / sample_size) * 100, "%", "complete", f"({index} / {sample_size})")
        print(f"MA_c_r2 = {sum(MA_c_r2) / len(MA_c_r2)}")
        print(f"MA_c_r3 = {sum(MA_c_r3) / len(MA_c_r3)}")
    
# final test forward pass
z1 = X[-1] * w1 + b1
r1 = ReLu(z1)
r2 = w2 * r1 + b2
r3 = w3 * r1 + b3
print(f"r2 = {r2}| y-true = {convert_to_one_hot(y[-1])[0]}")
print(f"r3 = {r3}| y-true = {convert_to_one_hot(y[-1])[1]}")

# store weights and biases in "stochastic_gradient_descent_raw_small_epoch_NN_data.txt" file
with open("stochastic_gradient_descent_raw_small_epoch_NN_data.txt", "w") as file:
    file.write(f"w1 = {w1}\nb1 = {b1}\nw2 = {w2}\nb2 = {b2}\nw3 = {w3}\nb3 = {b3}")


#optimal w and b

# w1 = 2.0
# b1 = -1.0
# w2 = 1.0
# b2 = 0.0
# w3 = -1.0
# b3 = 1.0