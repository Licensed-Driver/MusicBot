import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable
import math
import gzip
from numbers import Number
from NN_Definitions import Normalizer
    
class DenseLayer():

    """
    Version of the class initializer that is to be used when loading class information from a file.

    Args:

    """
    def __init__(self, input_shape:list[int]=None, out_channels:int=1, weights:torch.Tensor=None, biases:torch.Tensor=None, activation_func:Callable[[torch.Tensor],torch.Tensor]=None, activation_deriv_func:Callable[[torch.Tensor],torch.Tensor]=None, regularization_func:Callable[[torch.Tensor, torch.Tensor],torch.Tensor]=None, normalizer:Normalizer=None, dtype=torch.float):
        
        assert out_channels > 0, "Output channels must be greater than 0"
        assert (input_shape is not None) and (len(input_shape) > 2), "Must provide input shape that has at least [Batches, Channels, feature] dimensions (More dimensions for each new feature)."

        if ((activation_func is None) != (activation_deriv_func is None)):
            raise AssertionError("Must either, provide activation function AND derivative function, or provide neither.")
        elif (activation_func is None) and (activation_deriv_func is None):
            print("WARNING: Activation function is not set; defaulting to linear activation (Direct mapping).")
            self.activate:Callable[[torch.Tensor], torch.Tensor] = lambda x : x
            self.derivative:Callable[[torch.Tensor], torch.Tensor] = lambda x : torch.ones_like(x)
        else:
            self.activate = activation_func
            self.derivative = activation_deriv_func
            
        # Store a normalizer, if given, to normalize weighted values before activation
        self.normalizer = normalizer

        # A function that will allow for regularization based on the weights of this layer in backprop
        self.regularization_func = regularization_func

        # These next four are the only things that would require being saved to import a model
        if weights is not None:
            self.weights = weights
        else:
            self.weights = torch.randn([math.prod(input_shape[1:]), out_channels], dtype=dtype) /100   # Weight matrix of size [in_channels, out_channels]

        if biases is not None:
            self.biases = biases
        else:
            self.biases = torch.randn([out_channels], dtype=dtype)   # Bias matrix with one bias for each neuron
        # The pre and post activation outputs
        self.z:torch.Tensor = None
        self.a:torch.Tensor = None
        # Saving the activated output of the previous layer (input to this layer) for backprop
        self.input:torch.Tensor = torch.empty([0] + list(input_shape[1:]), dtype=dtype)
        # For saving the gradient for updating weights
        self.grad_w:torch.Tensor = None

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        self.input = x.view([x.shape[0], -1])   # Flatten the input tensor to only be [Batches, Features]

        self.z = Functional.clamp_add(Functional.clamp_matmul(self.input.cuda(), self.weights.cuda()), self.biases.cuda())  # Weight and bias the input

        # Normalize the weighted inputs before activation if using a normalizer
        if(self.normalizer is not None):
            z_hat = self.normalizer.normalize(x=self.z)
            self.a = self.activate(z_hat.cuda()).cpu()
        
        else:
            self.a = self.activate(self.z).cpu()  # Z already on cuda

        self.z = self.z.cpu()   # Move Z back to cpu after usage in activation

        return self.a
    
    """
    Uses the half delta provided to finish the delta calculation and compute gradients needed for weight updates.
    Returns the given delta matrix multiplied with it's weights, to be passed to the next layer
    """
    def backward(self, last_delta:torch.Tensor, step_size:float=1.0) -> torch.Tensor:

        # Calculate the delta by multiplying the past delta with the derivative of the tensor that was passed into the activation function
        if(self.normalizer is not None):
            delta = Functional.clamp_mul(x=last_delta.cuda(), y=self.derivative(self.normalizer.get_normalized()).cuda())
            delta = self.normalizer.backprop(self.z.cuda(), delta, step_size=step_size)
        else:
            delta = Functional.clamp_mul(x=last_delta.cuda(), y=self.derivative(self.z.cuda()).cuda())

        #print("Delta NaN:", last_delta.isnan().any(), "Input NaN:", self.input.isnan().any())
        self.grad_w = Functional.clamp_div(Functional.clamp_matmul(self.input.transpose(-1, -2).cuda(), delta), self.input.shape[0]).cpu()    # Not used until update so move to CPU

        # Allows for custom weight regularization and penalizing
        if self.regularization_func is not None:
            self.grad_w = self.regularization_func(self.grad_w.cuda(), self.weights.cuda()).cpu()

        # Update the bias since it isn't used in the gradient or delta calculations
        self.biases = Functional.clamp_sub(self.biases.cuda(), Functional.clamp_mul(delta.mean(dim=0), step_size)).cpu()    # Not used until forward so move to CPU

        # Set delta to be half ready to pass along
        delta = Functional.clamp_matmul(delta, self.weights.transpose(-1, -2).cuda())

        # Update the weights after the entire model has done backprop so that it's all updated using the non-updated weights for each layer
        #print("Dense Weights Before:", self.weights, "\nDense Grad:", torch.mul(step_size, self.grad_w.cuda()))
        clamped_grad:torch.Tensor = Functional.clamp_mul(self.grad_w.cuda(), step_size)
        self.weights = Functional.clamp_sub(self.weights.cuda(), clamped_grad.cuda()).cpu()
        #print("Dense Weights After:", self.weights)

        # Free up space
        del self.input, self.z, self.a, self.grad_w
        torch.cuda.empty_cache()

        return delta    # Used throughout the backprop on CUDA, so there'd no point in moving it back

    def get_out_shape(self) -> list[int]:
        return [1, self.weights.shape[1]]   # Batches isn't known yet, and it won't be used in any relevant calculations

    def to_dict(self) -> dict:
        return {
            'type':'dense',
            'biases':self.biases,
            'weights':self.weights,
            'input_shape':self.input.shape
            }
    
    def set_normalizer(self, normalizer:Normalizer):
        self.normalizer = normalizer
    
    @classmethod
    def from_dict(cls, import_dict:dict, activation_func:Callable[[torch.Tensor],torch.Tensor]=None, activation_deriv_func:Callable[[torch.Tensor],torch.Tensor]=None):
        return cls(input_shape=import_dict['input_shape'], weights=import_dict['weights'], biases=import_dict['biases'], activation_func=activation_func, activation_deriv_func=activation_deriv_func)


class ConvLayer():
    """
    Creates a convolutional layer that does forward and backward passes internally, including weight updates.

    Args:
        input_shape: Shape of the input Tensor (Requires input channels and batches, even if both are 1)
        out_channels: Number of output channels (this layer's neurons)
    """
    def __init__(self, out_channels:int=1, weights:torch.Tensor=None, biases:torch.Tensor=None, input_shape:list[int]=None, kernel_size:int=1, activation_func:Callable[[torch.Tensor],torch.Tensor]=None, activation_deriv_func:Callable[[torch.Tensor],torch.Tensor]=None, regularization_func:Callable[[torch.Tensor, torch.Tensor], torch.Tensor]=None, normalizer:Normalizer=None, dtype=torch.float):

        assert out_channels > 0, "Output channels must be greater than 0"
        assert kernel_size > 0, "Kernel size must be greater than 0."
        assert (input_shape is not None) and (len(input_shape) > 2), "Must provide input shape that has at least [Batches, Channels, feature] dimensions (More dimensions for each new feature)."

        # Make sure activation and derivative are given accordingly
        if (activation_func is None) and (activation_deriv_func is None):
            print("No activation given; defaulting to linear activation.")
            self.activate:Callable[[torch.Tensor], torch.Tensor] = lambda x : x
            self.derivative:Callable[[torch.Tensor], torch.Tensor] = lambda x : torch.ones_like(x)
        elif (activation_func is None) or (activation_deriv_func is None):
            raise AssertionError("Must either, provide activation function AND derivative function or provide neither.")
        else:
            self.activate = activation_func
            self.derivative = activation_deriv_func

        # Store a normalizer, if given, to normalize weighted values before activation
        self.normalizer = normalizer

        # A function that will allow for regularization based on the weights of this layer in backprop
        self.regularization_func = regularization_func

        # Store kernel_size and in_channels for future operations since kernel weights will stay transformed
        self.kernel_size = kernel_size
        self.input_info = {'shape':input_shape,
                           'padded_shape':None,
                           'stride':None,
                           'transformed_shape':None,
                           'transformed_stride':None,
                           'permuted_transformed_shape':None,
                           'permuted_transformed_stride':None}

        # Set the output shape to provide later for following layers
        self.output_shape = list(input_shape)
        self.output_shape[1] = out_channels

        # Store feature dimension to ensure the correct shape input is given in forward pass
        self.feature_dim = len(input_shape) - 2

        # Initiate random and null values
        if weights is not None:
            self.weights = weights
        else:
            self.weights = torch.randn([(kernel_size**(self.feature_dim))*input_shape[1], out_channels], dtype=dtype)/100 # Initiate weights as the kernel flattened to row vectors for matrix multiplication of [Channels, k^(feature_dim)*in_channels]
        if biases is not None:
            self.biases = biases
        else:
            self.biases = torch.randn([out_channels], dtype=dtype)   # Initiate one bias per filter since each filter uses independent weights, but each filter is one object
        self.z:torch.Tensor = None
        self.a:torch.Tensor = None
        self.input = torch.empty([0] + list(input_shape[1:]), dtype=dtype) # Create an empty tensor while ignoring batches, so we can use it's information without allocating as much memory
        # Save the expected return shape (without the batches as that is unknown)
        self.return_shape = list(self.input_info['shape'])
        self.return_shape[1] = out_channels # The return shape is going to be the same features with the in_channels changed to out_channels

    """
    Performs the forward pass with the current weights and biases.
    Stores the derivative of the activations for the backward pass.
    Args:
        x(Tensor): The input tensor of shape [Batches, in_channels, (Features)...]
    """
    def forward(self, x:torch.Tensor, kernel_stride:int=1) -> torch.Tensor:
        assert x.dim()-2 == self.feature_dim, "Input must be a tensor of shape [Batches, in_channels, features_h, features_w]."   # Make sure the input is the right amount of dimensions
        assert kernel_stride > 0, "stride must be greater than 0."

        # Get the shape of the non-padded input
        self.input_info['shape'] = x.shape

        # The size of the padding needs to be floor(k_size/2) on each side for keeping the shape of the original Tensor
        pad_size = self.kernel_size//2

        # Keep the batch and channel without padding
        padded_batch_channel_shape = list(x.shape[:2])
        # Each feature dimension needs to be padded with floor(k_size/2) zeros on all sides (times 2 to pad both sides of each dimension)
        padded_features_shape = [f_shape + (pad_size * 2) for f_shape in x.shape[2:]]

        # Create a padded Tensor of size [Batches, in_channels, D + pad_size*2...]
        # Where all new elements are on the edges and 0
        # Save to object since we need this shape input later in the back pass
        self.input = torch.zeros((padded_batch_channel_shape + padded_features_shape), device=x.device, dtype=x.dtype)
        # Create the index slices for setting x within the zero tensor (because we don't know the dimensions)
        index = tuple([slice(e) for e in padded_batch_channel_shape] + [slice(pad_size, e + pad_size) for e in x.shape[2:]])
        # Set the center to x
        self.input[index] = x

        # Keep the batches and channels the same
        batch_and_channel_shape = list(self.input.shape[:2])
        # We have a patch for every row column and every other feature dimension, at every stride (meaning we divide by stride)
        num_patches_shape = [(feature - self.kernel_size)//kernel_stride for feature in self.input.shape[2:]]
        # Shape of the kernel tensor for each patch
        patch_shape = [self.kernel_size] * len(self.input.shape[2:])

        # Keep batch and channel stride the same
        batch_channel_stride = list(self.input.stride()[:2])
        # Step through each feature dimension with the specified stride
        num_patches_stride = [p_stride * kernel_stride for p_stride in self.input.stride()[2:]]
        # Each patch is accessed like the original Tensor, since we are getting the convolution of each patch of the original Tensor
        patches_stride = list(self.input.stride()[2:])

        # Create a view of each patch at every valid stride point (any patches outside because of stride are not used)
        transformed_view = self.input.as_strided(
            # Assemble shape and stride
            size=(batch_and_channel_shape + num_patches_shape + patch_shape),
            stride=(batch_channel_stride + num_patches_stride + patches_stride)
            )
        
        # Store the stride and shape before permutation so that we can reference them directly when required
        self.input_info['transformed_shape'] = transformed_view.shape
        self.input_info['transformed_stride'] = transformed_view.stride()
        
        # Turn input from a 2d vector to flattened column patches that can be matrix multiplied by the kernel weights
        # Each column in each kernel-sized patch is converted into vertically stacked columns, and each patch is then stacked horizontally
        # Finally, we also flatten each channel to be stacked horizontally, so that we go from [Batches, in_channels, feature_dims...] to [Batches, feature_dims..., in_channels, kernel_dims...]
        transformed_view = transformed_view.permute([i for i in range(self.feature_dim + 2) if i != 1] + [1] + [i for i in range(self.feature_dim + 2, transformed_view.ndim)])  # Change dimensions to [Batches, feature_dims..., in_channels, kernel_dims...] for flattening
        
        # Set the input shape to reflect the real input shape (including batches which was uknown until now)
        self.input_info['padded_shape'] = self.input.shape
        # Save shape of transformed view for backprop unflattening (Done after permute since the dimensions have changed)
        self.input_info['permuted_transformed_shape'] = transformed_view.shape
        self.input_info['permuted_transformed_stride'] = transformed_view.stride()

        # Get the amount of patches by getting the product of each feature dimension and the channels
        flattened_patch_size = math.prod(transformed_view.shape[self.feature_dim+1:])
        # Get the size of each flattened 1D patch by getting the product of all kernel shapes and the in_channels
        num_patches = math.prod(transformed_view.shape[1:self.feature_dim+1])
        transformed_view = transformed_view.reshape(transformed_view.shape[0], num_patches, flattened_patch_size) # Reshape the Tensor with the new dimension order to flatten the correct dimensions into the shape we need

        self.input = transformed_view   # Set input to the patch version for backprop

        self.z = Functional.clamp_add(Functional.clamp_matmul(x=transformed_view.cuda(), y=self.weights.cuda()), self.biases.cuda())

        # Gets the shape that we are going to want our output to be to maintain consistency in normalizer and output shapes
        self.return_shape[0] = x.shape[0]

        # Normalize the weighted inputs before activation if using a normalizer
        if(self.normalizer is not None):
            z_hat = self.normalizer.normalize(x=self.z.transpose(-1, -2).contiguous().reshape(self.return_shape))
            self.a = self.activate(z_hat.cuda()).cpu()
        
        else:
            self.a = self.activate(self.z).cpu()  # Z already on cuda

        self.z = self.z.cpu()   # Send Z back to cpu after usage

        #print(self.a.shape, self.z.shape, self.weights.shape, self.input.shape, self.biases.shape)

        # reshape the return tensor to [Batches, out_channels, H, W]
        return self.a

    """
    Uses the half delta provided to finish the delta calculation and compute gradients needed for weight updates.
    Returns the given delta matrix multiplied with it's weights, to be passed to the next layer
    """
    def backward(self, last_delta:torch.Tensor, step_size:float=1.0) -> torch.Tensor:

        assert self.input_info['padded_shape'] is not None, "Cannot perform backpropagation without forward propagation first."

        # Reshape the delta back to the shape of the output
        delta = last_delta.contiguous().view(self.a.shape).cuda()
        intermediate_shape = list(self.z.shape)
        intermediate_shape[-1], intermediate_shape[-2] = intermediate_shape[-2], intermediate_shape[-1]
        delta = delta.reshape(intermediate_shape).transpose(-1, -2)

        # Calculate the delta by multiplying the past delta with the derivative of the tensor that was passed into the activation function
        if(self.normalizer is not None):
            # Get the intermediate shape that we have in the forward pass, since we need to exactly reverse the shaping steps for correct data locations
            delta = Functional.clamp_mul(x=delta, y=self.derivative(self.normalizer.get_normalized().reshape(intermediate_shape).transpose(-1, -2).cuda()))
            # Pass in the regular z shaped to be the expected working shape, not the internal shape
            delta = self.normalizer.backprop(self.z.transpose(-1, -2).contiguous().reshape(self.a.shape).cuda(), delta.transpose(-1, -2).reshape(self.a.shape), step_size=step_size)
            # Get the returned delta back to the internal z shape
            delta = delta.reshape(intermediate_shape).transpose(-1, -2)
        else:
            delta = Functional.clamp_mul(x=delta, y=self.derivative(self.z.cuda()).cuda())

        #print("Convo Input Delta:", last_delta.shape, "\nConvo Z:", self.z.shape, "\nNew Delta:", delta.shape, "\nConvo Weights:", self.weights.shape)

        # Calculate gradient as the mean of losses with respect to the weights
        self.grad_w = Functional.clamp_sum(Functional.clamp_div(x=Functional.clamp_matmul(x=self.input.transpose(-1, -2).cuda(), y=delta), y=self.input.shape[0]), dim=0, keepdim=False).cpu()   # Gradient won't be used until update, so move it to cpu

        # Allows for custom weight regularization and penalizing
        if self.regularization_func is not None:
            self.grad_w = self.regularization_func(self.grad_w.cuda(), self.weights.cuda()).cpu()

        # Update the bias with the average effect of each output dimension on the loss
        self.biases = Functional.clamp_sub(self.biases.cuda(), Functional.clamp_mul(delta.mean(dim=[0, 1]), step_size)).cpu()    # Update the bias since it isn't used in the gradient or delta calculations

        # Set delta to be half ready to pass along
        delta = Functional.clamp_matmul(x=delta, y=self.weights.transpose(-1, -2).cuda()).cpu()

        # Reshape the delta back into the shape of the transformed view that was used in the forward pass (unflatten the delta)
        delta = delta.reshape(shape=self.input_info['permuted_transformed_shape'])
        # Permute the dimensions to get the channels back to the second dimensions. We keep the batches at 0 and then just put everything past that, except for the channels, in order for all remaining dimensions
        channel_dim = self.feature_dim+1
        ndim = len(self.input_info['permuted_transformed_shape'])
        delta = delta.permute([0, channel_dim] + [i for i in range(1, ndim) if i != channel_dim])

        # Recreate the view of the input Tensor so that adding the delta will map each pixels delta to the correct pixel element
        input_grad = torch.zeros(self.input_info['padded_shape'])
        input_grad_view = input_grad.as_strided(
            size=delta.shape,
            stride=self.input_info['transformed_stride']    # Pre-permutation stride
        )

        # Maps the delta back to the correct shape in input_grad for backprop
        input_grad_view = Functional.clamp_add(input_grad_view, delta)

        # Crop the delta to get rid of the error of pad pixels
        pad_size = (self.input_info['padded_shape'][-1] - self.input_info['shape'][-1])//2
        index = tuple([slice(e) for e in input_grad.shape[:2]] + [slice(pad_size, e - pad_size) for e in self.input_info['padded_shape'][2:]])
        delta = input_grad[index]
        
        #print("Conv Weights Before:", self.weights, "\nConv Grad:", self.grad_w)
        clamped_grad:torch.Tensor = Functional.clamp_mul(self.grad_w.cuda(), step_size)
        self.weights = Functional.clamp_sub(self.weights.cuda(), clamped_grad.cuda()).cpu()
        #print("Conv Weights After:", self.weights)

        self.input_info['shape']=None
        self.input_info['padded_shape']=None
        self.input_info['transformed_shape']=None

        # Free up space
        del self.input, self.z, self.a, self.grad_w
        torch.cuda.empty_cache()

        return delta.contiguous()

    def get_out_shape(self) -> list[int]:
        return self.output_shape
    
    def to_dict(self) -> dict:
        return {
            'type':'conv',
            'weights':self.weights,
            'biases':self.biases,
            'input_shape':self.input_info['shape'],
            'kernel_size':self.kernel_size
        }
    
    def set_normalizer(self, normalizer:Normalizer):
        self.normalizer = normalizer
    
    @classmethod
    def from_dict(cls, import_dict:dict, activation_func:Callable[[torch.Tensor],torch.Tensor]=None, activation_deriv_func:Callable[[torch.Tensor],torch.Tensor]=None):
        return cls(input_shape=import_dict['input_shape'], weights=import_dict['weights'], biases=import_dict['biases'], kernel_size=import_dict['kernel_size'], activation_func=activation_func, activation_deriv_func=activation_deriv_func)

"""
A dynamic neural network architecture that allows for procedural layer generation.
The first layer added will always be considered the input layer, and must match the input dimensions.
The last layer added will always be the output layer, but at least one layer must be added after the initial input layer. (This would cause a direct mapping.)
"""
class NN():
    """
    Initiates the first layer of the network and creates the framework to add more later.
    """
    def __init__(self, input_shape:list[int]=[1,1,1], dtype:torch.Type=torch.float):
        self.layers:list[DenseLayer|ConvLayer] = []
        self.prev_shape = input_shape
        self.dtype = dtype
        self.weight_averages:list[torch.Tensor] = []

    def sum_weights(self) -> torch.Tensor:
        weights=torch.zeros(1)
        for layer in self.layers:
            weights=Functional.clamp_add(weights, layer.weights.sum())
        return weights

    def sum_abs_weights(self) -> torch.Tensor:
        weights=torch.zeros(1)
        for layer in self.layers:
            weights=Functional.clamp_add(weights, layer.weights.abs().sum())
        return weights

    def gen_fc(self, out_channels:int, activation_func:Callable[[torch.Tensor], torch.Tensor], activation_deriv_func:Callable[[torch.Tensor], torch.Tensor], regularization_func:Callable[[torch.Tensor, torch.Tensor],torch.Tensor]=None, normalizer:Normalizer=None):

        self.layers.append(DenseLayer(input_shape=self.prev_shape, out_channels=out_channels, activation_func=activation_func, activation_deriv_func=activation_deriv_func, regularization_func=regularization_func, dtype=self.dtype))
        self.prev_shape = self.layers[-1].get_out_shape()
        if(normalizer is not None):
            self.layers[-1].set_normalizer(normalizer=normalizer(self.prev_shape))

        self.weight_averages.append(torch.zeros(self.layers[-1].weights.shape))

    def gen_con(self, out_channels:int=None, kernel_size:int=None, activation_func:Callable[[torch.Tensor],torch.Tensor]=None, activation_deriv_func:Callable[[torch.Tensor],torch.Tensor]=None, regularization_func:Callable[[torch.Tensor, torch.Tensor],torch.Tensor]=None, normalizer:Normalizer=None):
        assert (len(self.layers) == 0) or (not isinstance(self.layers[-1], DenseLayer)), "Cannot use a convolutional layer after the data has been flattened to a dense layer."

        self.layers.append(ConvLayer(input_shape=self.prev_shape, out_channels=out_channels, kernel_size=kernel_size, activation_func=activation_func, activation_deriv_func=activation_deriv_func, regularization_func=regularization_func, dtype=self.dtype))
        self.prev_shape = self.layers[-1].get_out_shape()
        if(normalizer is not None):
            self.layers[-1].set_normalizer(normalizer=normalizer(self.prev_shape))

        self.weight_averages.append(torch.zeros(self.layers[-1].weights.shape))

    def export_model(self, filepath:str):
        with gzip.open(filepath, 'wb') as file:
            torch.save([layer.to_dict() for layer in self.layers], file)

    def import_model(self, filepath:str, activation_func:Callable[[torch.Tensor], torch.Tensor], activation_deriv_func:Callable[[torch.Tensor], torch.Tensor]):
        with gzip.open(filepath, 'rb') as file:
            loaded = torch.load(file)
        
        for layer in loaded:
            if layer['type'] == 'dense':
                self.layers.append(DenseLayer.from_dict(layer, activation_func=activation_func, activation_deriv_func=activation_deriv_func))
            elif layer['type'] == 'conv':
                self.layers.append(ConvLayer.from_dict(layer, activation_func=activation_func, activation_deriv_func=activation_deriv_func))

    def train(self, x:torch.Tensor, y:torch.Tensor, epochs:int=1, step_size:float=1.0, step_func:Callable[[float], float]=None, loss_deriv_func:Callable[[torch.Tensor, torch.Tensor], torch.Tensor]=None):
        if step_func is None:
            print("WARNING: Step size function was not specified, therefore the step size will not change between epochs.")
        if loss_deriv_func is None:
            print("WARNING: Loss function was not provided, therefore a standard Mean Squared Error (MSE) loss function will be applied.")
            loss_deriv_func=Functional.mse

        SWA_epochs = math.floor(0.75*epochs)
        for i in range(5, 0, -1):
            SWA_interval = math.floor((epochs-SWA_epochs)/i)
            if(SWA_interval != 0): break
        SWA_snapshots = 0
    
        for _ in range(epochs):

            print(f"Starting Epoch #{_}")
            
            input = x

            for layer in self.layers:
                input = layer.forward(input)

            #print("SOFTMAX OUTPUT",layer.a, "\nSUM",layer.a.sum(dim=1), "UNACTIVATED:", layer.z)

            percentage = (input.argmax(dim=1)==y.argmax(dim=1)).sum().item()/len(input.argmax(dim=1))
            print(f"Epoch #{_+1} Accuracy: {round(percentage*100, 2)}%")

            delta = loss_deriv_func(input.cuda(), y.cuda()).cpu()

            for layer in reversed(self.layers):
                delta = layer.backward(delta, step_size=step_size)

            if(_ >= SWA_epochs):
                for i in range(len(self.layers)):
                    SWA_snapshots += 1
                    self.weight_averages[i] = Functional.clamp_div(Functional.clamp_add(Functional.clamp_mul(self.weight_averages[i], SWA_snapshots - 1), self.layers[i].weights), SWA_snapshots)
                
                SWA_epochs += SWA_interval

            del delta
            torch.cuda.empty_cache()

            if(step_func is not None):
                step_size = step_func(step_size)

            print(f"Epoch #{_+1} Completed")

        for i in range(len(self.layers)):
            self.layers[i].weights = self.weight_averages[i]

        print(f"{epochs} Epochs completed.")

    def predict(self, x:torch.Tensor) -> torch.Tensor:
        input = x

        for layer in self.layers:
            input = layer.forward(input)

        return input

    def get_model(self) -> list[DenseLayer|ConvLayer]:
        return

class Functional():

    class Batch_Normalizer(Normalizer):

        def __init__(self, input_shape:list[int]):
            self.gamma = torch.ones(size=([1, input_shape[1]]+[1 for _ in input_shape[2:]]))    # The scaling values
            self.beta = torch.zeros(size=self.gamma.shape) # The shifting values
            self.normalized = torch.tensor([])   # The saved input for backprop
            self.std = torch.tensor([])    # Saved varaince for derivative
            self.mean = torch.tensor([])    # Saved mean for derivative
        
        def normalize(self, x:torch.Tensor) -> torch.Tensor:
            epsilon = 1e-6  # A small constant to avoid divide by zero errors
            if(self.gamma.device != x.device):
                self.gamma = self.gamma.to(x.device)
            if(self.beta.device != x.device):
                self.beta = self.beta.to(x.device)

            self.mean = Functional.batch_mean(x)
            self.std = Functional.clamp_add(Functional.batch_std(x), epsilon)

            self.normalized = Functional.clamp_div(Functional.clamp_sub(x, self.mean), self.std)

            ret = Functional.clamp_add(Functional.clamp_mul(self.gamma, self.normalized), self.beta)

            return ret
        
        def get_normalized(self):
            return self.normalized

        def backprop(self, z:torch.Tensor, delta:torch.Tensor, step_size:float=1) -> torch.Tensor:

            if(self.gamma.device != delta.device):
                self.gamma = self.gamma.to(delta.device)
            if(self.beta.device != delta.device):
                self.beta = self.beta.to(delta.device)
            if(self.normalized.device != delta.device):
                self.normalized = self.normalized.to(delta.device)
            if(self.std.device != delta.device):
                self.std = self.std.to(delta.device)
            if(self.mean.device != delta.device):
                self.mean = self.mean.to(delta.device)

            # Get number of elements per channel for the normalized derivative
            elements_per_channel = math.prod([self.normalized.shape[i] for i in range(self.normalized.ndim) if i != 1])    # Get total elements per channel for derivs

            # Calculate derivative of the loss with respect to the gamma and the beta for updating
            beta_deriv = Functional.clamp_sum(delta, dim=[i for i in range(delta.dim()) if i != 1], keepdim=True)
            gamma_deriv = Functional.clamp_sum(Functional.clamp_mul(delta, self.normalized), dim=[i for i in range(delta.dim()) if i != 1], keepdim=True)

            normalized_deriv = Functional.clamp_mul(delta, self.gamma)  # This is the dx(hat)/dx

            # Calculate all elements per channel summed for use in equations
            summed_delta = Functional.clamp_sum(delta, dim=[i for i in range(delta.dim()) if i != 1], keepdim=True)
            # Calculate the delta of normalized z minus the summed delta of norm z over number of elements (delta_z - (1/m)*sum(delta_z))
            regular_deriv_partone = Functional.clamp_sub(normalized_deriv, Functional.clamp_div(summed_delta, elements_per_channel))
            # Calculate ((z-mean)/(variance+epsilon)) * (summed_delta/elements_per_channel)
            regular_deriv_parttwo = Functional.clamp_mul(Functional.clamp_div(Functional.clamp_sub(z, self.mean), Functional.clamp_mul(self.std, self.std)), Functional.clamp_div(Functional.clamp_sum(Functional.clamp_mul(delta, Functional.clamp_sub(z, self.mean)), dim=[i for i in range(z.dim()) if i != 1], keepdim=True), elements_per_channel))
            # Calculate the final equation
            regular_deriv= Functional.clamp_div(Functional.clamp_sub(regular_deriv_partone, regular_deriv_parttwo), self.std)
            
            self.beta = Functional.clamp_sub(self.beta, Functional.clamp_mul(beta_deriv, step_size))
            self.gamma = Functional.clamp_sub(self.gamma, Functional.clamp_mul(gamma_deriv, step_size))
            return regular_deriv

    @staticmethod
    def mse(x:torch.Tensor, y:torch.Tensor) -> float:
        diff = (x-y)
        # Return 2 times the loss in all batches
        return Functional.clamp_mul(2, diff)
    
    """
    Takes in a tensor with assumed shape [Batches, Out Channels, Feature Dimensions...] and returns the mean of each channel accross all batches.
    Args:
        x: Tensor of shape [Batches, Out Channels, Feature Dimensions...]
    Returns:
        channel mean: Tensor of shape [1, Out Channels] (First dimension left as 1 for casting)
    """
    @staticmethod
    def batch_mean(x:torch.Tensor) -> torch.Tensor:
        return Functional.clamp_mean(x, dim=[i for i in range(x.dim()) if i != 1], keepdim=True)
    
    """
    Takes in a tensor with assumed shape [Batches, Out Channels, Feature Dimensions...] and returns the standard deviation of each channel accross all batches.
    Args:
        x: Tensor of shape [Batches, Out Channels, Feature Dimensions...]
    Returns:
        channel std: Tensor of shape [1, Out Channels] (First dimension left as 1 for casting)
    """
    @staticmethod
    def batch_std(x:torch.Tensor) -> torch.Tensor:
        return Functional.clamp_std(x, dim=[i for i in range(x.dim()) if i != 1], unbiased=False, keepdim=True)
    
    @staticmethod
    def clamp_div(x:torch.Tensor, y:torch.Tensor|Number, min:Number=None, max:Number=None) -> torch.Tensor:
        finfo = torch.finfo(x.dtype)
        if(max is None):
            max=finfo.max
        if(min is None):
            min=finfo.min
        return torch.div(x, y).clamp(min=min, max=max)

    @staticmethod
    def clamp_mul(x:torch.Tensor, y:torch.Tensor|Number, min:Number=None, max:Number=None) -> torch.Tensor:
        finfo = torch.finfo(x.dtype)
        if(max is None):
            max=finfo.max
        if(min is None):
            min=finfo.min
        return torch.mul(x, y).clamp(min=min, max=max)
    
    @staticmethod
    def clamp_std(x:torch.Tensor, keepdim:bool=False, dim:int|list[int]=None, unbiased:bool=False, min:Number=None, max:Number=None) -> torch.Tensor:
        finfo = torch.finfo(x.dtype)
        if(max is None):
            max=finfo.max
        if(min is None):
            min=finfo.min
        if(dim is None):
            dim=[i for i in range(x.dim())]
        return x.std(dim=dim, keepdim=keepdim, unbiased=unbiased).clamp(min=min, max=max)
    
    @staticmethod
    def clamp_mean(x:torch.Tensor, keepdim:bool=False, dim:int|list[int]=None, min:Number=None, max:Number=None) -> torch.Tensor:
        finfo = torch.finfo(x.dtype)
        if(max is None):
            max=finfo.max
        if(min is None):
            min=finfo.min
        if(dim is None):
            dim=[i for i in range(x.dim())]
        return x.mean(dim=dim, keepdim=keepdim).clamp(min=min, max=max)
    
    @staticmethod
    def clamp_matmul(x:torch.Tensor, y:torch.Tensor, min:Number=None, max:Number=None) -> torch.Tensor:
        finfo = torch.finfo(x.dtype)
        if(max is None):
            max=finfo.max
        if(min is None):
            min=finfo.min
        return torch.matmul(x, y).clamp(min=min, max=max)
    
    @staticmethod
    def clamp_sum(x:torch.Tensor, dim:int|list[int]=0, keepdim:bool=False, min:Number=None, max:Number=None) -> torch.Tensor:
        finfo = torch.finfo(x.dtype)
        if(max is None):
            max=finfo.max
        if(min is None):
            min=finfo.min
        return torch.sum(input=x, dim=dim, keepdim=keepdim).clamp(min=min,max=max)
    
    @staticmethod
    def clamp_add(x:torch.Tensor, y:torch.Tensor|Number, min:Number=None, max:Number=None) -> torch.Tensor:
        finfo = torch.finfo(x.dtype)
        if(max is None):
            max=finfo.max
        if(min is None):
            min=finfo.min
        return torch.add(x, y).clamp(min=min,max=max)

    @staticmethod
    def clamp_sub(x:torch.Tensor, y:torch.Tensor|Number, min:Number=None, max:Number=None) -> torch.Tensor:
        finfo = torch.finfo(x.dtype)
        if(max is None):
            max=finfo.max
        if(min is None):
            min=finfo.min
        return torch.sub(x, y).clamp(min=min,max=max)
    
    @staticmethod
    def clamp_exp(x:torch.Tensor, min:Number=None, max:Number=None) -> torch.Tensor:
        finfo = torch.finfo(x.dtype)
        if(max is None):
            max=finfo.max
        if(min is None):
            min=finfo.min
        return torch.exp(x).clamp(min=min,max=max)
    
    @staticmethod
    def clamp_tanh(x:torch.Tensor, min:Number=None, max:Number=None) -> torch.Tensor:
        finfo = torch.finfo(x.dtype)
        if(max is None):
            max=finfo.max
        if(min is None):
            min=finfo.min
        return torch.tanh(x).clamp(min=min,max=max)
    
    @staticmethod
    def leaky(x):
        return torch.where(x>=0, x, 0.01*x)

    @staticmethod
    def leakyDeriv(x):
        x.cuda()
        return torch.where(x >= 0, torch.ones_like(x), torch.full_like(x, 0.01))

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linearDeriv(x):
        return torch.ones_like(x)

    @staticmethod
    def sigmoid(x:torch.Tensor, dim=1):
        ret = Functional.clamp_exp(x.neg())
        ret = Functional.clamp_add(ret, 1)
        ret = Functional.clamp_div(1, ret)
        return ret
    
    @staticmethod
    def softmax(x:torch.Tensor, dim=1, gamma=1e-6):
        exp = Functional.clamp_exp(x)
        ret = Functional.clamp_add(Functional.clamp_sum(exp, dim=dim, keepdim=True), gamma)
        ret = Functional.clamp_div(exp, ret)
        return ret
    
    @staticmethod
    def softmax_cross_entropy_deriv(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        return x - y