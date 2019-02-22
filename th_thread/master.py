import torch
from torch.autograd import Variable as V
import torch.distributed as dist
from torch import nn as nn
import argparse
from torch.multiprocessing import Process, Queue, Value, Event
from multiprocessing.managers import BaseManager as bm
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

epoch_event = Event()
global_event = Event()
save_event = Event()

grad_queue = Queue(args.buffer)

targets_queue = Queue(args.buffer)

def get_epoch_event():
    return epoch_event

def get_global_event():
    return global_event

def get_save_event():
    return save_event

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', help='the ip of manager server', default='89.72.2.41')
    parser.add_argument('-path', help='the path of share system')
    parser.add_argument('-buffer', type=int, help='the size of queue', default=2)
    args = parser.parse_args()
    if os.path.exists(args.path):
        os.remove(args.path)

    bm.register('get_epoch_event', callable=get_epoch_event)
    bm.register('get_global_event', callable=get_global_event)
    bm.register('get_grad_queue', callable=lambda: grad_queue)
    bm.register('get_targets_queue', callable=lambda: targets_queue)
    bm.register('get_save_event', callable=lambda: save_event)
    m = bm(address=(args.ip, 5000), authkey=b'xpipe')
    m.start()
    g_e = m.get_global_event()
    print("master run......")
    g_e.wait()
    m.shutdown()
    print("master shutdown........")
