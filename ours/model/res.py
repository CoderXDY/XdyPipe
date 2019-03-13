import torch
from torch.autograd import Variable as V
import torch.distributed as dist
from torch import nn as nn
import argparse
from torch.multiprocessing import Process, Queue, Value
import torch.nn.functional as F
import torch.optim as optim
import logging as Log
import time


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResInputLayer(nn.Module):
    def __init__(self):
        super(ResInputLayer, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out

class ResBlockLayer(nn.Module):
    def __init__(self, block, planes, num_blocks, stride, in_planes=None):
        super(ResBlockLayer, self).__init__()
        if in_planes is not None:
            self.in_planes = in_planes
        else:
            self.in_planes = 64
        self.layer = self._make_layer(block, planes, num_blocks, stride)

    def _make_layer(self, block, planes, num_blocks, stride):

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        return out

    def get_in_plances(self):
        return self.in_planes



class ResOutputLayer(nn.Module):

    def __init__(self, block, num_classes=10):
        super(ResOutputLayer, self).__init__()
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        out = F.avg_pool2d(x, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


"""

ResNet18 splits two groups to develop in Th-2

"""
class THResNetGroup0(nn.Module):
    def __init__(self):
        super(THResNetGroup0, self).__init__()
        self.layer0 = ResInputLayer()
        self.layer1 = ResBlockLayer(BasicBlock, 64, 2, 1, 64)
        self.layer2 = ResBlockLayer(BasicBlock, 128, 2, 2, 64)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        return out


class THResNetGroup1(nn.Module):
    def __init__(self):
        super(THResNetGroup1, self).__init__()
        self.layer0 = ResBlockLayer(BasicBlock, 256, 2, 2, 128)
        self.layer1 = ResBlockLayer(BasicBlock, 512, 2, 2, 256)
        self.layer2 = ResOutputLayer(BasicBlock)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

"""

ResNet50 splits two groups to develop in Th-2

in_plance: 64
in_plance: 256
in_plance: 512
in_plance: 1024
in_plance: 2048
torch.Size([1, 256, 32, 32])
torch.Size([1, 512, 16, 16])
torch.Size([1, 1024, 8, 8])
torch.Size([1, 2048, 4, 4])
torch.Size([1, 10])

"""
class THResNet50Group0(nn.Module):
    def __init__(self):
        super(THResNet50Group0, self).__init__()
        self.layer0 = ResInputLayer()
        self.layer1 = ResBlockLayer(Bottleneck, 64, 3, 1, 64)
        self.layer2 = ResBlockLayer(Bottleneck, 128, 4, 2, 256)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        return out


class THResNet50Group1(nn.Module):
    def __init__(self):
        super(THResNet50Group1, self).__init__()
        self.layer0 = ResBlockLayer(Bottleneck, 256, 6, 2, 512)
        self.layer1 = ResBlockLayer(Bottleneck, 512, 3, 2, 1024)
        self.layer2 = ResOutputLayer(Bottleneck)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        return out



"""
3-nodes for ResNet 101


in_plance: 64
in_plance: 256
in_plance: 512
in_plance: 1024
in_plance: 2048
torch.Size([1, 256, 32, 32])
torch.Size([1, 512, 16, 16])
torch.Size([1, 1024, 8, 8])
torch.Size([1, 2048, 4, 4])
torch.Size([1, 10])
"""
class THResNet101Group0(nn.Module):
    def __init__(self):
        super(THResNet101Group0, self).__init__()
        self.layer0 = ResInputLayer()
        self.layer1 = ResBlockLayer(Bottleneck, 64, 3, 1, 64)
        self.layer2 = ResBlockLayer(Bottleneck, 128, 4, 2, 256)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

class THResNet101Group1(nn.Module):
    def __init__(self):
        super(THResNet101Group1, self).__init__()
        self.layer = ResBlockLayer(Bottleneck, 256, 13, 2, 512)

    def forward(self, x):
        out = self.layer(x)
        return out

class THResNet101Group2(nn.Module):
    def __init__(self):
        super(THResNet101Group2, self).__init__()
        self.layer0 = ResBlockLayer(Bottleneck, 256, 10, 1, 1024)
        self.layer1 = ResBlockLayer(Bottleneck, 512, 3, 2, 1024)
        self.layer2 = ResOutputLayer(Bottleneck)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        return out








if __name__ == '__main__':
    group0 = THResNet101Group0()
    group1 = THResNet101Group1()
    group2 = THResNet101Group2()
    x = torch.randn(1, 3, 32, 32)
    x = group0(x)
    print("group 0:" + str(x.size()))
    x = group1(x)
    print("group 1: " + str(x.size()))
    y = group2(x)
    print(y.size())








