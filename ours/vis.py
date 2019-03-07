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
parser.add_argument('--train', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()


path_list = args.file.split()
path_result = {}
for path in path_list:
    loss_list = []
    with open(path, 'r') as f:
        for line in f:
            strs = line.strip().split(':')
            if args.train and strs[0] == 'train':
                loss_list.append(strs[1])
            elif strs[0] == 'eval':
                loss_list.append(strs[1])
    path_result[path] = loss_list

vis = visdom.Visom()

x = list(range(600))

stack = []
for name, result in path_result.items():
    stack.append(np.array(result))


vis.line(X=x, Y=np.column_stack(stack), win='loss', opts=dict(showlegend=True))