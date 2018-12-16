import torch
from torch.autograd import Variable as V
import torch.distributed as dist
from torch import nn as nn
import argparse
from torch.multiprocessing import Process, Queue, Value, Event
import torch.nn.functional as F
import torch.optim as optim
import logging as Log
import time
from queue import Empty
import traceback

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
    print("rank-" + str(dist.get_rank()) + ".....")
    if dist.get_rank() == 0:
        send_opt = dist.isend(tensor=torch.zeros(1), dst=1)
        send_opt.wait()
        print("rank " + str(dist.get_rank()) + " has send...")
    elif dist.get_rank() == 1:
        try:
            rev_val = torch.randn(1)
            rev_opt = dist.irecv(tensor=rev_val, src=0)
            rev_opt.wait()
            print("666")
        except RuntimeError as error:
            #print("rank " + str(dist.get_rank()) + "has recive...")
            print("rank 1 rev_val is None")
            #traceback.format_exc(error)
        finally:
            print("forever")







def init_processes(fn, path, size, rank, q, e):
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
        p = Process(target=init_processes, args=(run, args.path, args.size, rank, q, e))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
