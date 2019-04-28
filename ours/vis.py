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
import torch.backends.cudnn as cudnn
from visdom import Visdom
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('-file', help='the filename of log')
parser.add_argument('-count', type=int)
parser.add_argument('-key')
parser.add_argument('-prop', type=int)
args = parser.parse_args()

path_list = args.file.split()
path_result = {}
for path in path_list:
    plots = []
    count = 0
    index = 0
    with open(path, 'r') as f:
        for line in f:
            if count == args.count:
                break
            strs = line.strip().split(':')
            if strs[0] == args.key:
                if index % args.prop == 0:
                    plots.append(float(strs[1]))
                    if count < 20:
                        print(path + " " + str(float(strs[1])))
                    count += 1
                    index += 1
                else:
                    index += 1
            else:
                pass
        path_result[path] = plots

vis = Visdom()

x = list(range(args.count))
print(len(x))
stack = []
for name, result in path_result.items():
    stack.append(np.array(result))
    print(len(result))
vis.line(X=x, Y=np.column_stack(stack), win='loss', opts=dict(showlegend=True))

"""
for path in path_list:
    loss_list = []
    count = 0
    with open(path, 'r') as f:
        for line in f:
            if count == args.count:
                break
            strs = line.strip().split(':')
            if args.train and strs[0] == 'train':
                if float(strs[1]) > 2:
                    print(strs[1])
                loss_list.append(float(strs[1]))
                count += 1
            elif not args.train and strs[0] == 'eval':
                print(strs[0])
                loss_list.append(float(strs[1]))
                count += 1
    path_result[path] = loss_list

vis = Visdom()

x = list(range(args.count))

stack = []
for name, result in path_result.items():
    stack.append(np.array(result))


vis.line(X=x, Y=np.column_stack(stack), win='loss', opts=dict(showlegend=True))
"""
