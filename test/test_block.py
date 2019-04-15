import torch
import torch.distributed as dist
from torch import nn as nn
import argparse
from torch.multiprocessing import Queue, Event, Process
from multiprocessing.managers import BaseManager as bm
#from multiprocessing.dummy import Process
from multiprocessing.dummy import Queue as ThreadQueue
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing
import logging
import time
import torchvision
import torchvision.transforms as transforms

import traceback
from queue import Empty, Full
import os
import psutil
import gc
import numpy as np
"""
 pipeline ResNet script for Tianhe-2  with gpu cluster

"""



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-flag', type=int, help='the size of queue', default=2)
    args = parser.parse_args()

    bm.register('get_queue')
    m = bm(address=('127.0.0.1', 5000), authkey=b'xpipe')
    m.connect()
    queue = m.get_queue()
    if args.flag == 0:
        for i in range(10):
            #value = np.random.rand(2,2)
            value = torch.randn([128, 6, 6, 64])
            #print(value)
            print("----")
            queue.put(value)
            time.sleep(1)
    else:
        for i in range(10):
            value = queue.get()
            #print(value)
            print("----")



