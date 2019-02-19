from torch import nn as nn
import argparse
import torch.nn.functional as F
import torch.optim as optim
import logging
import time
import torchvision
import torchvision.transforms as transforms
from utils import progress_bar
import traceback
from queue import Empty, Full
import os
import psutil
import gc
from resnet import ResNet18
from resnet152_dist import ResNet50
from resnet_pipe_model import ResPipeNet18, ResPipeNet50
import torch.backends.cudnn as cudnn
from visdom import Visdom
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('-file', help='the filename of log')
args = parser.parse_args()


path_list = args.file.split()
name_loss = {}
for path in path_list:
    losss = []
    with open(path, 'r') as f:
        for line in f:
            strs = f.strip().split('--')
            loss = strs[1].split(":")[1]
            losss.append(loss)
    name_loss[path] = losss

vis = visdom.Visom()

x = list(range(200))

stack = []
for name, loss in name_loss.items():
    stack.append(np.array(loss))


vis.line(X=x, Y=np.column_stack(stack), win='loss', opts=dict(showlegend=True))