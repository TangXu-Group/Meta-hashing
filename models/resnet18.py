import torch.nn as nn
from utils.Self_Adaptive_Conv import SAPConv
import torch


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 downsample=False):
        super(BasicBlock, self).__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.down = downsample

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(self.in_channels,
                               self.out_channels,
                               self.kernel_size,
                               self.stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.out_channels,
                               self.out_channels,
                               self.kernel_size,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        if self.down:
            self.downsample = nn.Sequential(
                nn.Conv2d(self.in_channels,
                          self.out_channels,
                          kernel_size=1,
                          stride=self.stride),
                nn.BatchNorm2d(self.out_channels))

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MultiBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(MultiBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(self.in_channels,
                               self.out_channels,
                               self.kernel_size,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SAPConv(self.out_channels,
                             self.out_channels,
                             is_bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Resnet18(nn.Module):
    def __init__(self, K_bits=1000):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # yapf:disable
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, kernel_size=3),
            BasicBlock(64, 64, kernel_size=3))
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, kernel_size=3, stride=2, downsample=True),
            BasicBlock(128, 128, kernel_size=3))
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, kernel_size=3, stride=2, downsample=True),
            BasicBlock(256, 256, kernel_size=3))
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, kernel_size=3, stride=2, downsample=True),
            MultiBlock(512, 512, kernel_size=3))
        # MultiBlock(512, 512, kernel_size=3))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, K_bits))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)

        features = self.avgpool(x4)
        features = features.reshape(features.size(0), -1)
        real_hash = self.fc(features)
        return real_hash
