import torch
import torch.nn as nn


class MSE_Quantization(nn.Module):

    def __init__(self):
        super(MSE_Quantization, self).__init__()

    def forward(self, real_hash):
        ideal_hash = torch.sign(real_hash)
        loss = torch.pow(real_hash - ideal_hash, 2)
        return loss.mean()


class Bitwise_Quantization(nn.Module):

    def __init__(self):
        super(Bitwise_Quantization, self).__init__()

    def forward(self, inputs):
        loss = -(inputs * torch.log(inputs + 1e-20) +
                 (1.0 - inputs) * torch.log(1.0 - inputs + 1e-20))
        return loss.mean()
