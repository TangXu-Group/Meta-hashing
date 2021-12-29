import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import math
import torch.nn.backends


class SAPConv(nn.Module):
    def __init__(self, in_channels, out_channels, is_bias=True):
        super(SAPConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_bias = is_bias
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, 3, 3))
        if self.is_bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        conv1 = F.conv2d(x,
                         weight=self.weight,
                         bias=self.bias,
                         stride=1,
                         padding=1,
                         dilation=1)
        conv1 = F.relu(conv1)
        conv2 = F.conv2d(x,
                         weight=self.weight,
                         bias=self.bias,
                         stride=1,
                         padding=2,
                         dilation=2)
        conv2 = F.relu(conv2)
        conv3 = F.conv2d(x,
                         weight=self.weight,
                         bias=self.bias,
                         stride=1,
                         padding=3,
                         dilation=3)
        conv3 = F.relu(conv3)
        # fusion
        conv1_sum = torch.sum(conv1, dim=1, keepdim=True)
        conv2_sum = torch.sum(conv2, dim=1, keepdim=True)
        conv3_sum = torch.sum(conv3, dim=1, keepdim=True)
        attention = torch.cat([conv1_sum, conv2_sum, conv3_sum], dim=1)

        # find the global min value
        min_value = torch.min(attention)
        max_value = torch.max(attention)

        attention = (attention - min_value + 1e-20) / \
            (max_value - min_value + 1e-10)
        enhanced_attention = -1 / (torch.log2(attention) - 1e-20)
        # enhanced_attention = torch.exp(enhanced_attention)
        if torch.min(enhanced_attention) < 0:
            print("The value within attention is error!!")
        softmax = F.softmax(enhanced_attention, dim=1)
        lambda1, lambda2, lambda3 = softmax.split(1, dim=1)
        return conv1.mul(lambda1) + conv2.mul(lambda2) + conv3.mul(lambda3)
