from abc import ABC, abstractmethod
import torch
import math
from numbers import Number

class Normalizer(ABC):

    @abstractmethod
    def __init__(self, input_shape:list[int]):
        pass

    @abstractmethod
    def normalize(self, x:torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_normalized(self) -> torch.Tensor:
        pass

    @abstractmethod
    def backprop(self, delta:torch.Tensor, step_size:float=1.0) -> torch.Tensor:
        pass