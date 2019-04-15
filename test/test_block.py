import torch
import torch.distributed as dist
from torch import nn as nn
import argparse
import torch.distributed as dist
from torch.multiprocessing import Process
from multiprocessing.dummy import thread
from multiprocessing.dummy import Queue as ThreadQueue
import torch.nn.functional as F
import torch.optim as optim
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




def run(rank, size):

    def backward():
        pass

    if rank == 0:
        back_process = thread(target=backward)
    elif rank == 1:
        pass

def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()



