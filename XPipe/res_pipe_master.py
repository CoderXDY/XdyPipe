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


class Acc(object):
    def __init__(self):
        self.global_acc = 0.0
        self.best_acc = 0.0

    def set_global_acc(self, value):
        self.global_acc = value

    def set_best_acc(self, value):
        self.best_acc = value

    def get_global_acc(self):
        return self.global_acc

    def get_best_acc(self):
        return self.best_acc


def get_epoch_event():
    return epoch_event

def get_global_event():
    return global_event


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', help='the ip of manager server', default='89.72.2.41')
    parser.add_argument('-path', help='the path of share system')
    args = parser.parse_args()
    if os.path.exists(args.path):
        os.remove(args.path)
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    bm.register('get_epoch_event', callable=get_epoch_event)
    bm.register('get_global_event', callable=get_global_event)
    bm.register('get_acc', Acc)
    m = bm(address=(args.ip, 5000), authkey=b'xpipe')
    m.start()
    g_e = m.get_global_event()
    print("master run......")
    g_e.wait()
    m.shutdown()
    print("master shutdown........")
