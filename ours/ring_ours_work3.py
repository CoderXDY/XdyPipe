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
from model.res import THResNet101Group0, THResNet101Group2, THResNet101Group1
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


def get_left_right(rank, size):

    return left, right


def transfer(rank, send_buf, shape):
    left, right = get_left_right(rank)
    send_opt = dist.isend(tensor=send_buf, dst=right)
    send_opt.wait()
    recv_buf = torch.zeros(shape, dtype=torch.int8)
    return recv_buf


def parallel(layer, trainloader, rank):
    criterion.cuda()
    outputs_queue = ThreadQueue(args.buffer_size)
    if rank == 0:
        back_process = Process(target=backward_rank0, args=())
        back_process.start()
    elif rank == 1:
        back_process = Process(target=backward_rank1, args=())
        back_process.start()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if rank == 0:
            inputs = inputs.cuda()
            outputs = layer(inputs)
            outputs_queue.put(outputs)
        elif rank == 1:

        elif rank == 2:







def train():
    parallel()
