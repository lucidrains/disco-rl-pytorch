import torch
from torch import nn
from torch.nn import Module

from x_transformers import Decoder, Encoder

class DiscoRL(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
