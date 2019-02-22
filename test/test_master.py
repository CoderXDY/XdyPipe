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
import torchvision
import torchvision.transforms as transforms

import traceback
from queue import Empty
import os
import numpy
global_event = Event()

def get_global_event():
    return global_event
if __name__ == "__main__":

    queue = Queue(10)
    bm.register('get_queue', callable=lambda: queue)
    bm.register('get_global_event', callable=get_global_event)
    m = bm(address=('127.0.0.1', 5000), authkey=b'xpipe')
    m.start()
    g_e = m.get_global_event()
    print("master run......")
    g_e.wait()
    m.shutdown()
    print("master shutdown........")

