from random import uniform

X = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
y = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]

class NeuralNetwork:
    def __init__(self):
        # <activation_function> is a function that will be used in the <Neuron> class
        # missing data for <number_of_neurons>, <activation_function>
        self.input_layer = InputLayer(1)
        self.hidden_layers = [Layer(number_of_neurons, activation_function), Layer(number_of_neurons, activation_function)]
        self.output_layer = Layer(number_of_neurons, activation_function)
        print("NeuralNetwork initialized!")

class Layer:
    def __init__(self, number_of_neurons, activation_function):
        self.number_of_neurons = number_of_neurons
        # missing data for <number_of_inputs>, <number_of_outputs>
        self.neurons = [Neuron(number_of_inputs) for neuron_index in range(number_of_neurons)]
        self.activation_function = activation_function
        print("Layer initialized!")

class InputLayer(Layer):
    def __init__(self, number_of_input_neurons):
        
        def activation_function(self, result):
            return result
        
        self.number_of_neurons = number_of_neurons
        # missing data for <number_of_inputs>, <number_of_outputs>
        self.neurons = [Neuron(number_of_inputs, number_of_outputs) for neuron_index in range(number_of_neurons)]
        self.activation_function = activation_function
        print("Layer initialized!")
        
        super().__init__(self, number_of_input_neurons, activation_function)
        print("InputLayer initialized!")

class Neuron:
    def __init__(self, number_of_inputs):
        self.number_of_inputs = number_of_inputs
        self.weights = [Weight() for input_index in range(number_of_inputs)]
        self.bias = Bias()
        print("Neuron initialized!")

    def forward(self, input):
        def forward_propagation_function(self, input):
            forward_result = None
            for weight in self.weights:
                forward_result += weight.value * input
            forward_result += self.bias.value
            return forward_result
                
        def activation_function(self, result):
            activated_result = self.activation_function(result)
            return activated_result
                
        result = self.forward_propagation_function(input)
        activated_result = self.activation_function(result)
        return activated_result
            

class Weight:
    def __init__(self):
        self.value = uniform(-1.0, 1.0)


class Bias:
    def __init__(self):
        self.value = uniform(-1.0, 1.0)

            
    
    class ForwardPropagate:
        def __init__(self, layers):
            self.layers = layers
        

