import torch
import torch.distributed as dist
from torch import nn as nn
import argparse

from torch.multiprocessing import Queue, Event
from multiprocessing.managers import BaseManager as bm
from multiprocessing.dummy import Process
from multiprocessing.dummy import Queue as ThreadQueue
from multiprocessing.dummy import Semaphore
import torch.nn.functional as F
import torch.optim as optim
import logging
import time
from model.res import THResNet101Group40, THResNet101Group41, THResNet101Group42, THResNet101Group43
from model.vgg_module import VggLayer
from model.googlenet import GoogleNetGroup0, GoogleNetGroup1, GoogleNetGroup2
from model.dpn import  THDPNGroup0, THDPNGroup1, THDPNGroup2
import torchvision
import torchvision.transforms as transforms
from utils import progress_bar
import traceback
from queue import Empty, Full
import os
import psutil
import gc
import torch.backends.cudnn as cudnn
import numpy as np
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='the path fo share file system')
    parser.add_argument('-type', help='size of batch')
    parser.add_argument('-rank', type=int, default=2)
    args = parser.parse_args()
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/' + args.model + '-' + args.type + '-rank-' + str(args.rank) + '_ckpt.t7')
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print("best_acc: " + str(best_acc))
    print("start_epoch: " + str(start_epoch))