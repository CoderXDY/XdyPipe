import torch
from torch.autograd import Variable as V
import torch.distributed as dist
from torch import nn as nn
import argparse
from torch.multiprocessing import Process, Queue, Value, Event
from multiprocessing.dummy import Process as TProcess
import torch.nn.functional as F
import torch.optim as optim
import logging as Log
import time
from queue import Empty
import traceback
import threading
"""



test some trait and code

.....
use queue comunication.....

"""



Log.basicConfig(filename='pipeline2.log', level=Log.INFO)

torch.manual_seed(10)
class Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(Layer, self).__init__()
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer(x))
        return x




# def run(q, e):
#
#     print("rank-" + str(dist.get_rank()) + "......")
#     if dist.get_rank() == 0:
#         for i in range(5):
#             q.put(torch.randn(1))
#         e.wait()
#     elif dist.get_rank() == 1:
#         while True:
#             print("get.....")
#             try:
#                 t = q.get(block=True, timeout=3)
#                 print(t.size())
#             except Empty as empty:
#                 traceback.format_exc()
#                 print("empty")
#                 e.set()
#                 break

def run(q, e):


    if dist.get_rank() == 0:
        print(threading.currentThread().name)
        print("rank-" + str(dist.get_rank()) + ".....")
        opt = dist.irecv(tensor=torch.randn(1), src=1)
        opt.wait()
        print("rank-" + str(dist.get_rank()) + " success receve")
    elif dist.get_rank() == 1:
        print(threading.currentThread().name)
        print("rank-" + str(dist.get_rank()) + ".....")
        opt = dist.isend(tensor=torch.randn(1), dst=0)
        opt.wait()
        print("rank-" + str(dist.get_rank()) + " success send")






def init_processes(fn, path, size, rank, q, e):
    print("init-" + str(rank))
    dist.init_process_group(backend='tcp', init_method=path, world_size=size, rank=rank)
    fn(q, e)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-size', type=int, help='input the sum of node')
    parser.add_argument('-path', help='the path fo share file system')
    args = parser.parse_args()
    processes = []
    print("size:" + str(args.size))
    print("path:" + args.path)

    q = Queue(10)
    e = Event()
    for rank in range(args.size):
        print(rank)
        p = Process(target=init_processes, args=(run, args.path, args.size, rank, q, e))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
