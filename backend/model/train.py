import architecture as NN
import torch

class Trainer():
    """
    Creates a trainer to train and test based on the given datasets.
    Args:
        x_train(ndarray): Training fetures
        x_test(ndarray): Testing features
        y_train(ndarray): Training labels
        y_test(ndarray): Testing labels
    """
    def __init__(self, x_train, x_test, y_train, y_test, architecture):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.net = architecture

    def train(self, epochs:int=100, initial_step:float=1):
        for t in range(epochs):
            self.net.forward()
            self.net.backward(self.y_train, initial_step)
    
    def test(self, loss_func):
        return loss_func(self.net.predict(self.x_test), self.y_test)