import random 
import math
from autodiff import Value
class Neuron: 
    def __init__(self, n_inputs, activation='relu'): #n_inputs is number of inputs to specific neuron
        self.w = [Value(random.uniform(-1,1)) for _ in range (n_inputs)]
        self.b = Value(random.uniform(-1,1))
        self.activation = activation

    def __call__(self,x): # Calls n(x) where x is data and n is a neuron 
        weighted_sum = sum(w_i * x_i for w_i, x_i in zip(self.w, x)) + self.b # w*x + b

        if self.activation == 'tanh':
            activation = weighted_sum.tanh()
        elif self.activation == 'sigmoid':
            activation = weighted_sum.sigmoid()
        else:
            activation = weighted_sum.relu() # Do relu activation unless the user specifies

        return activation

    def parameters(self):
        return self.w + [self.b] # All parameters for that neuron

class Layer:
    def __init__ (self, n_inputs, n_outputs, activation = "relu"): # n_outputs is the number of neurons in the layer 
        self.neurons = [Neuron(n_inputs, activation) for _ in range (n_outputs)]
    def __call__(self, x): 
        outputs = [n(x) for n in self.neurons]
        return outputs
    def parameters(self):
        parameters = [] 
        for neuron in self.neurons:
            ps = neuron.parameters() 
            parameters.extend(ps) 
        return parameters 

class MLP: 
    def __init__(self, n_inputs, n_outputs, activation = "relu"): # n_outputs is a list of the sizes of each layer
        layer_size = [n_inputs] + n_outputs # Concatenate the input layer to the list of other layers
        self.layers = []
        for i in range(len(n_outputs)): # layer_size is of size n_outputs+1
            self.layers.append(Layer(layer_size[i], layer_size[i+1], activation))

    def __call__(self, input): # input is input/data vector x
        for layer in self.layers:
            x = layer(input)
        return x
    
    def parameters(self):
        parameters = [] 
        for layer in self.layers:
            ps = layer.parameters() 
            parameters.extend(ps) 
        return parameters 