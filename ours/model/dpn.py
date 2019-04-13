import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
        super(Bottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        self.conv1 = nn.Conv2d(last_planes, in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv3 = nn.Conv2d(in_planes, out_planes+dense_depth, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes+dense_depth)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv2d(last_planes, out_planes+dense_depth, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes+dense_depth)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        x = self.shortcut(x)
        d = self.out_planes
        out = torch.cat([x[:,:d,:,:]+out[:,:d,:,:], x[:,d:,:,:], out[:,d:,:,:]], 1)
        out = F.relu(out)
        return out




class DPNInputLayer(nn.Module):
    def __init__(self):
        super(DPNInputLayer, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out

class DPNBlockLayer(nn.Module):
    def __init__(self, cfg):
        super(DPNBlockLayer, self).__init__()
        in_planes, out_planes, num_blocks, dense_depth, self.last_planes, stride = cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[5]
        self.layer = self._make_layer(in_planes, out_planes, num_blocks, dense_depth, stride)

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i,stride in enumerate(strides):
            layers.append(Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i==0))
            self.last_planes = out_planes + (i+2) * dense_depth
        return nn.Sequential(*layers)

    def forward(self, out):
        out = self.layer(out)
        return out

class DPNOutputLayer(nn.Module):
    def __init__(self, out_planes, num_blocks, dense_depth):
        super(DPNOutputLayer, self).__init__()
        self.linear = nn.Linear(out_planes + (num_blocks + 1) * dense_depth, 10)

    def forward(self, out):
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class THDPNGroup0(nn.Module):
    def __init__(self):
        super(THDPNGroup0, self).__init__()
        self.input_layer = DPNInputLayer()
        layer_cfg0 = [96, 256, 3, 16, 64, 1]
        layer_cfg1 = [192, 512, 4, 32, 320, 2]
        self.layer0 = DPNBlockLayer(layer_cfg0)
        self.layer1 = DPNBlockLayer(layer_cfg1)
    def forward(self, x):
        out = self.input_layer(x)
        out = self.layer0(out)
        out = self.layer1(out)
        return out

class THDPNGroup1(nn.Module):
    def __init__(self):
        super(THDPNGroup1, self).__init__()
        layer_cfg2 = [384, 1024, 20, 24, 672, 2]
        self.layer = DPNBlockLayer(layer_cfg2)
    def forward(self, x):
        out = self.layer(x)
        return out


class THDPNGroup2(nn.Module):
    def __init__(self):
        super(THDPNGroup2, self).__init__()
        layer_cfg3 = [768, 2048, 3, 128, 1528, 2]
        self.layer0 = DPNBlockLayer(layer_cfg3)
        self.layer1 = DPNOutputLayer(layer_cfg3[1], layer_cfg3[2], layer_cfg3[3])
    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        return out


"""
dpn92
torch.Size([1, 672, 16, 16])
torch.Size([1, 1528, 8, 8])
torch.Size([1, 10])


"""


def test():
    group0 = THDPNGroup0()
    group1 = THDPNGroup1()
    group2 = THDPNGroup2()
    x = torch.randn(1, 3, 32, 32)
    y = group0(x)
    print(y.size())
    y = group1(y)
    print(y.size())
    y = group2(y)
    print(y.size())
