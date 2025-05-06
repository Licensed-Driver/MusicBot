import architecture as arch
import torch
from keras.api.datasets import mnist 

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

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
train_y = train_y.reshape(train_y.shape[0], 1)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])
test_y = test_y.reshape(test_y.shape[0], 1)
testNet = arch.NeuralNet(input=testTensor, input_dim=train_x.shape[1], output_dim=1, hidden_layers=3, samples=train_x.shape[0], neurons=[10, 10, 10], layer_activators=[leaky, leaky, torch.sigmoid], layer_derivs=[leakyDeriv, leakyDeriv, sigmoidDeriv])


#testNet.debug()
testNet.forward()
testNet.debug()
testNet.backward(testOutTensor, 0.000001)
testNet.debug()
for i in range(1000):
    testNet.forward()
    testNet.backward(testOutTensor, 0.000001)
print("FINAL")
print(testNet.output())



