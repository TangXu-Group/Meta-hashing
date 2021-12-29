import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import torch.nn.init as init
import torch.nn.functional as F
from utils.Self_Adaptive_Conv import SAPConv


class ClsNet(nn.Module):
    def __init__(self, K_bits=24, num_classes=21):
        super(ClsNet, self).__init__()
        self.logits = nn.Sequential(
            nn.Linear(K_bits, num_classes)
        )

    def forward(self, x):
        logits = self.logits(x)
        return logits


class MetaNet(nn.Module):

    def __init__(self, K_bits=24):
        super(MetaNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            SAPConv(256, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, K_bits),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
