import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable

class NeuralNet():
    def __init__(self, input:torch.Tensor=None, input_dim=0, output_dim=1, samples=1, hidden_layers=0, neurons:list[int]=None, layer_activators:list[Callable[[torch.Tensor], torch.Tensor]]=None, layer_derivs:list[Callable[[torch.Tensor], torch.Tensor]]=None, loss_deriv:Callable[[torch.Tensor, torch.Tensor], torch.Tensor]=None, dtype=torch.float32):

        assert isinstance(layer_activators, list) or (layer_activators is None), "layer_activators must be a list or None."
        assert isinstance(layer_derivs, list) or (layer_derivs is None), "layer_derivs must be a list or None."
        if((layer_derivs is not None) and (layer_activators is None)) or ((layer_derivs is None) and (layer_activators is not None)):
            raise SyntaxError("layer_derivs and layer_activaotrs must both be set, if using custom activation functions.")
        elif not (len(layer_derivs) == len(layer_activators)):
            raise SyntaxError("layer_derivs and layer_activators must have the same length to use custom activation functions.")
        assert hidden_layers >= 0
        assert samples >= 1
        assert input_dim or input, "If input is not provided, input dimension must be specified."

        assert isinstance(neurons, list), "Neurons must be a list."
        if(neurons is None): neurons=[input_dim for _ in range(hidden_layers)]
        if(neurons is not None) and (len(neurons) < hidden_layers):
            print("WARNING: Neurons specified in neurons list is less than hidden layers. The last number of neurons specified will be used for the remaining layers.")
            for _ in range(len(neurons), hidden_layers):
                neurons.append(neurons[-1])
        elif(neurons is not None) and (len(neurons) > hidden_layers):
            print("WARNING: Number of neurons specified is greater than the number of hidden layers. Only the neurons from index 0 to index hidden_layers-1 will be used")
        self.input = torch.cat([torch.ones([1, samples]).to('cuda'), input]).to('cuda') if(input is not None) else torch.zeros([input_dim + 1, samples], dtype=dtype).to('cuda')  # Input tensor with plus one for bias
        self.input_dim = self.input.shape[0]-1  # Set the input dimension for comparison later without considering biases
        self.samples = samples

        if(loss_deriv is None):
            print("WARNING: No loss derivative specified, so the default of MSE will be used.")
            self.loss_deriv = MSE_deriv

        # Function Prototypes
        self.leakyRelu = None

        # If we want hidden layers then we create a list of hidden layers and initialize it fully connected
        if(hidden_layers):
            self.weights = [torch.randn([neurons[i-1]+1, neurons[i]]).cuda() for i in range(1, hidden_layers)] # Create a tensor of each weight matrix excluding the last one and one more column for the biases
            self.weights.insert(0, torch.randn([input_dim+1, neurons[0]]).cuda())  # Tensor for the first weights to multiply the input plus 1 for biases
            self.weights.append(torch.randn([neurons[-1]+1, output_dim]).cuda()) # The last weight matrix that takes in a hidden activation output and matrix multiplies into the output dimension plus one column for the biases
            self.deep = True    # To check if we need to consider deep layers later
            self.outputs = [torch.zeros([samples, neurons[i]+1]).cuda() for i in range(hidden_layers)]  # Create a tensor for all intermediate values with one more column for the biases
            self.outputs.append(torch.zeros([samples, output_dim]).cuda())  # Set the last entry to the final outputs

        else:
            self.weights = [torch.randn([input_dim+1, output_dim]).to('cuda')]    # The first and last weight matrix in the case that there are no hidden layers
            self.deep = False

        if(layer_activators is not None) and (len(layer_activators) < len(self.weights)):
            print("WARNING: There are less layer activators that weights (hidden layers + 1 output layer). The last activator will be used to activate all missing layers.")
            self.activate = layer_activators
            self.derivs = layer_derivs
            for i in range(len(layer_activators), len(self.weights)):
                self.activate.append(self.activate[-1])
                self.derivs.append(self.derivs[-1])
        elif(layer_activators is not None) and (len(layer_activators) > len(self.weights)):
            print("WARNING: There are more layer activators that weights (hidden layers + 1 output layer). The activators will only be used until the last layer.")
            self.activate = layer_activators[:self.weights.__len__()-1]
            self.derivs = layer_derivs[:self.weights.__len__()-1]
        else:
            self.activate = layer_activators
            self.derivs = layer_derivs
    
    """
    Sets the input matrix of the network object.
    Args:
        x(torch.Tensor || numpy.ndarray): Input matrix [features, sample]
    Returns:
        N/A
    """
    def setInput(self, x):
        assert x.shape[0]==self.input_dim, "Input dimension must match the dimension specified on object creation."
        # Set the input matrix to x with an appended bias term to each sample to save time in forward propogation
        if(isinstance(x, np.ndarray)): self.input = torch.cat([torch.ones([1, self.input.shape[1]]), torch.from_numpy(x)], dim=0).to('cuda')

        elif(isinstance((x, torch.Tensor))): self.input = torch.cat([torch.ones([1, self.input.shape[1]]), x]).to('cuda')

        else: raise TypeError("x must be of either np.ndarray or torch.Tensor type.")

    """
    Prints the weights, inputs, and intermediate outputs for debugging.
    """
    def debug(self):
        print("INPUTS:", self.input,"\n","OUTPUTS:",self.outputs,"\n","WEIGHTS:",self.weights,"\n")

    def output(self):
        return self.outputs[-1].tolist()

    """
    Uses ReLU for forward propagation with the provided input.
    SUGGESTION: Try not to pass the input here, as it will have to set the input as a new cuda tensor with every iteration of the forward pass
    WARNING: If input matrix was not given as a parameter nor on creation, and no input matrix has been set (using this.setInput) then a zero matrix of shape [input_dim, samples] will be used.
    Args:
        (optional) x(torch.Tensor.to('cuda')): If input is directly provided, self.input will be overwritten (Must be of shape [input_dim, any])
    Returns:
        N/A
    """
    def forward(self, x:torch.Tensor=None):

        if(x is not None):
            assert isinstance(x, torch.Tensor), "Input matrix must be torch.Tensor type."
            assert x.shape[0]==self.input_dim, "x dimension must match input diomension specified at object creation"
            forwardIn = torch.cat([torch.ones([self.input.shape[1], 1]).to('cuda'), x], dim=0).to('cuda')
        else:
            forwardIn = self.input

        if self.deep:

            activation = torch.matmul(forwardIn.T, self.weights[0])
            #activation = (activation - activation.mean(dim=1, keepdim=True))/(activation.std(dim=1, keepdim=True) + 1e-8)
            activation = self.activate[0](activation)
            print(activation)
            self.outputs[0] = torch.cat([torch.ones([activation.shape[0], 1]).to('cuda'), activation], dim=1)

            # Multiplies all intermediate deep layers up to but not including the last layer
            for layer in range(1, self.outputs.__len__()):
                activation = torch.matmul(self.outputs[layer-1], self.weights[layer])   # Adding one bias column to each intermediate matrix during the weight multiplication
                #activation = (activation - activation.mean(dim=1, keepdim=True))/(activation.std(dim=1, keepdim=True) + 1e-8)
                activation = self.activate[layer](activation)
                self.outputs[layer] = torch.cat([torch.ones([activation.shape[0], 1]).to('cuda'), activation], dim=1)
        else:
            self.outputs[-1] = torch.matmul(forwardIn.T, self.weights[-1])

    """
    Uses MSE for a first order gradient descent, updating it's own weights
    Args:
        y(torch.Tensor): Labels
        step_size: Uhhhh I feel like you should know this one
    Returns:
        N/A
    """
    def backward(self, y, step_size):

        # Gets the derivative of each activated layer
        output_derivs = [self.derivs[i](self.outputs[i][:, 1:]) for i in range(len(self.outputs))]

        # Calculate initial delta as the gradient of the loss function with respect to the output times the derivative of the output activation
        delta = torch.mul(self.loss_deriv(self.outputs[-1][:, 1], y), output_derivs[-1])

        for layer in range(1, len(self.weights)):
            # Calculate gradient of the loss with respect to the current layer's weight
            grad_w = torch.matmul(self.outputs[(-layer)-1].T, delta)
            # Update the weight
            self.weights[-layer] = self.weights[-layer] - step_size * grad_w
            # Calculate the delta term of this layer which is the derivative of the output function of with respect to the weights of the last layer
            delta = torch.mul(torch.matmul(delta, self.weights[-layer][1:, :].T), output_derivs[(-layer)-1])
        # Calculate the gradient of the input weights
        grad_w = torch.matmul(self.input, delta)
        self.weights[0] = self.weights[0] - step_size * grad_w
    
    def MSE_deriv(self, x, y):
        return (2/x.shape[0]) * (x[-1][:, 1:] - y.T)

"""
A dynamic neural network architecture that allows for procedural layer generation.
The first layer added will always be considered the input layer, and must match the input dimensions.
The last layer added will always be the output layer, but at least one layer must be added after the initial input layer. (This would cause a direct mapping.)
"""
class NN():
    """
    Initiates all relevant lists and values, creating only the space allocation for the input layer.
    """
    def __init__(self, input_shape:list[int]=[1,1], bias:bool=True):
        self.input = torch.zeros(input_shape).cuda()
        self.layers:list[torch.Tensor] = [] # Holds the activated value of each layer (i.e. the value of each neuron)
        self.layer_derivs:list[torch.Tensor] = []   # Holds the derivative of each layers activation during each forward pass
        self.weights:list[torch.Tensor] = []    # Holds the weights for all layers past the input
        self.activations:list[Callable[[torch.Tensor],]] = []    # Holds references to all the activation functions (in-place)
        self.derivatives:list[Callable[[torch.Tensor], torch.Tensor]] = []   # Holds the references to all the activtion function derivatives (not in-place)
        self.layer_types:list[int] = [] # Tell us what type each layer is so that we know when to flatten (0=fc, 1=con)
        self.bias=bias  # Whether we include the bias terms in each layer

    def gen_fc_layer(self, neurons, activation_func, activation_deriv_func):
        self.layers.append(torch.zeros([]))
        self.layer_derivs.append(torch.zeros([]))
        self.activations.append(activation_func)
        self.derivatives.append(activation_deriv_func)
        self.weights.append(self.layers)

    def gen_con_layer(self, )