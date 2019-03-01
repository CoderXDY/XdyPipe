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
    return {'read_point': Value('i', 0), 'write_point': Value('i', 0), 'read_signal': Event(), 'write_signal': Event()}, \
           [torch.zeros(shape).cuda(cuda_id).share_memeory_() for i in range(size)]









def put_tensor(queue, atom, tensor):
    if (atom['write_point'].value + 1) % len(queue) == atom['read_point'].value:
        atom['write_signal'].wait()
    queue[atom['write_point'].value].copy_(tensor)
    val = atom['write_point'].value
    atom['write_point'].value = (val + 1) % len(queue)
    atom['write_signal'].clear()
    atom['read_signal'].set()




def get_tensor(queue, atom):
    length = len(queue)
    if atom['write_point'].value == atom['read_point'].value:
        atom['read_signal'].wait()
    val = atom['read_point'].value
    if val + 1 == length:
        atom['read_point'].value = (val + 1) % length
        return queue[length - 1]
    atom['read_point'].value = val + 1
    return queue[atom['read_point'].value - 1]
    atom['read_signal'].clear()
    atom['write_signal'].set()


def read(queue, atom):
    while True:
        x = get_tensor(queue, atom)
        print(x)
def write(queue, atom):
    for i  in range(8):
        x = torch.FloatTensor([i])
        put_tensor(queue, atom, x)
        time.sleep(1)


if __name__ == "__main__":
    atom, queue = get_tensor_queue(4, [2], 0)
    f_p = Process(target=read, args=(queue, atom))
    f_p.start()
    b_p = Process(target=write,
                  args=(queue, atom))
    b_p.start()
    f_p.join()
    b_p.join()