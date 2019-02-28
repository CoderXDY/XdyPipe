import torch
import torch.distributed as dist
from torch import nn as nn
import argparse

from torch.multiprocessing import Queue, Event, Value
from multiprocessing.managers import BaseManager as bm
from multiprocessing.dummy import Process
from multiprocessing.dummy import Queue as ThreadQueue
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing
import logging
import time
from res import THResNetGroup0, THResNetGroup1
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

#n.grad_fn.next_functions[1][0].next_functions[0][0] 0.4942, 0.3581
def get_tensor_queue(size, shape, cuda_id):
    return {'read_point': Value('i', 0), 'write_point': Value('i', 0), 'signal': Event()} , \
           [torch.zeros(shape).cuda(cuda_id).share_memeory_() for i in range(size)]

def put_tensor(queue, tensor, atom):
    if atom['write_point'].value % len(queue) == atom['read_point'].value:
        atom['signal'].wait()
    queue[atom['write_point'].value].copy_(tensor)
    queue[atom['write_point'].value]