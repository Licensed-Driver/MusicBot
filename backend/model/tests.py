import architecture as arch
import torch
from keras.api.datasets import mnist
import math
import numpy as np
from functools import partial
from numbers import Number

# These are just some very basic unit tests to identify architecture problems during development

testTensor = torch.Tensor([[3],[3]]).to('cuda')
testOutTensor = torch.Tensor([[1238],[4328]]).to('cuda')
def leaky(x):
    return torch.where(x>=0, x, 0.01*x)

def leakyDeriv(x):
    x.cuda()
    return torch.where(x >= 0, torch.ones_like(x), torch.full_like(x, 0.01))

def sigmoidDeriv(x):
    x.cuda()
    return torch.mul(x, (x-1))

def linear(x):
    return x

def linearDeriv(x):
    return torch.ones_like(x)
def softmax(x:torch.Tensor, dim=1, gamma=1e-6):
    exp = arch.Functional.clamp_exp(x)
    ret = arch.Functional.clamp_add(arch.Functional.clamp_sum(exp, dim=dim, keepdim=True), gamma)
    ret = arch.Functional.clamp_div(exp, ret)
    return ret

def softmax_cross_entropy_deriv(x:torch.Tensor, y:torch.Tensor):
    return x - y
def soft_cross_lasso_deriv(NN:arch.NN, x:torch.Tensor, y:torch.Tensor, gamma:float=None):
    if(gamma is not None):
        weight_sums = NN.sum_abs_weights()

        lasso_term = gamma*weight_sums
        softmax_result = arch.Functional.clamp_sub(softmax(x, dim=1), y)
        lasso_result = arch.Functional.clamp_add(softmax_result, lasso_term)
        return lasso_result
    else:
        return softmax_cross_entropy_deriv(x=x, y=y)
def lasso_regularization(gamma:Number, gradient:torch.Tensor, weight:torch.Tensor) -> torch.Tensor:
    return arch.Functional.clamp_add(gradient, arch.Functional.clamp_mul(weight.sign(), gamma))

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = torch.tensor(train_x).unsqueeze(1)
train_x = train_x / 255.0
train_y = torch.nn.functional.one_hot(torch.from_numpy(train_y).long(), num_classes=10)
test_x = torch.tensor(test_x).unsqueeze(1)
test_x = test_x / 255.0
test_y = torch.nn.functional.one_hot(torch.from_numpy(test_y).long(), num_classes=10)

testNet = arch.NN(train_x.shape, dtype=torch.float)

testNet.gen_con(out_channels=5, kernel_size=4, activation_func=leaky, activation_deriv_func=leakyDeriv, normalizer=arch.Functional.Batch_Normalizer)
testNet.gen_fc(out_channels=10, activation_func=softmax, activation_deriv_func=linearDeriv, normalizer=arch.Functional.Batch_Normalizer)

epoch_count = 5

testNet.train(x=train_x[:100], y=train_y[:100], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[100:200], y=train_y[100:200], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[200:300], y=train_y[200:300], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[300:400], y=train_y[300:400], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[400:500], y=train_y[400:500], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[500:600], y=train_y[500:600], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[600:700], y=train_y[600:700], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[700:800], y=train_y[700:800], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[800:900], y=train_y[800:900], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[900:1000], y=train_y[900:1000], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[1000:1100], y=train_y[1000:1100], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[1100:1200], y=train_y[1100:1200], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[1200:1300], y=train_y[1200:1300], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[1300:1400], y=train_y[1300:1400], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[1400:1500], y=train_y[1400:1500], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[:100], y=train_y[:100], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[100:200], y=train_y[100:200], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[200:300], y=train_y[200:300], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[300:400], y=train_y[300:400], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[400:500], y=train_y[400:500], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[500:600], y=train_y[500:600], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[600:700], y=train_y[600:700], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[700:800], y=train_y[700:800], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[800:900], y=train_y[800:900], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[900:1000], y=train_y[900:1000], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[1000:1100], y=train_y[1000:1100], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[1100:1200], y=train_y[1100:1200], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[1200:1300], y=train_y[1200:1300], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[1300:1400], y=train_y[1300:1400], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[1400:1500], y=train_y[1400:1500], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[:100], y=train_y[:100], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[100:200], y=train_y[100:200], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[200:300], y=train_y[200:300], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[300:400], y=train_y[300:400], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[400:500], y=train_y[400:500], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[500:600], y=train_y[500:600], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[600:700], y=train_y[600:700], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[700:800], y=train_y[700:800], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[800:900], y=train_y[800:900], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[900:1000], y=train_y[900:1000], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[1000:1100], y=train_y[1000:1100], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[1100:1200], y=train_y[1100:1200], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[1200:1300], y=train_y[1200:1300], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[1300:1400], y=train_y[1300:1400], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)
testNet.train(x=train_x[1400:1500], y=train_y[1400:1500], epochs=epoch_count, step_size=0.1, loss_deriv_func=softmax_cross_entropy_deriv)

prediction = (testNet.predict(test_x[:100]))

matches = prediction.argmax(dim=1) == test_y[:100].argmax(dim=1)
percentage = (matches.sum().item()/len(prediction.argmax(dim=1))) * 100
print("Prediction:", prediction.argmax(dim=1), "\nActual:", test_y[:100].argmax(dim=1), f"\nAccuracy: {round(percentage, 2)}%")
print("SUCCESS")

