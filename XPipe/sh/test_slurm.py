import torch
from torch.autograd import Variable as V
import torch.distributed as dist
from torch import nn as nn
import argparse
from torch.multiprocessing import Process, Queue, Value, Event
import torch.nn.functional as F
import torch.optim as optim
import logging
import time
from res import BasicBlock, Bottleneck, ResInputLayer, ResBlockLayer, ResOutputLayer
import traceback
from queue import Empty
import socket



def run():

    print("rank-" + str(dist.get_rank()) + " start........")
    time.sleep(10)
    print("rank-" + str(dist.get_rank()) + " end........")



def init_processes(fn, args, rank):
    myname = socket.getfqdn(socket.gethostname())
    print("node: " + myname + " rank----" + str(rank))
    dist.init_process_group(backend='tcp', init_method=args.path, world_size=args.size, rank=rank)
    fn()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-size', type=int, help='input the sum of node')
    parser.add_argument('-rank', type=int, help='rank')
    parser.add_argument('-path', help='the path fo share file system')
    args = parser.parse_args()

    p0 = Process(target=init_processes, args=(run, args, args.rank))
    p1 = Process(target=init_processes, args=(run, args, (11 - args.rank)))
    p0.start()
    p1.start()