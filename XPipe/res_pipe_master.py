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

e = Event()
target_buffer = Queue(20)

def get_event():
    return e

def get_queue():
    return target_buffer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', help='the ip of manager server', default='89.72.2.41')
    parser.add_argument('-path', help='the path of share system')
    args = parser.parse_args()
    if os.path.exists(args.path):
        os.remove(args.path)

    bm.register('get_event', callable=get_event)
    bm.register('get_queue', callable=get_queue)
    m = bm(address=(args.ip, 5000), authkey=b'xpipe')
    m.start()
    m_e = m.get_event()
    print("master run......")
    m_e.wait()
    m.shutdown()
    print("master shutdown........")
