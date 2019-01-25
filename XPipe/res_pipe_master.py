import torch
from torch.autograd import Variable as V
import torch.distributed as dist
from torch import nn as nn
import argparse
from torch.multiprocessing import Process, Queue, Value, Event
from torch.multiprocessing import BaseManager as bm
import torch.nn.functional as F
import torch.optim as optim
import logging
import time
from res import BasicBlock, Bottleneck, ResInputLayer, ResBlockLayer, ResOutputLayer
import torchvision
import torchvision.transforms as transforms
from utils import progress_bar
import traceback
from queue import Empty
import os

e = Event()

def get_event():
    return e
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', help='the ip of manager server')
    args = parser.parse_args()
    bm.register('get_event', callable=get_event)
    m = bm(address=(args.ip, 5000), authkey='xpipe')
    
    m.start()
