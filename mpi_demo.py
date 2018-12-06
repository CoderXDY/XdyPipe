import torch
from torch.autograd import Variable as V
import torch.distributed as dist
from torch import nn as nn
import argparse
from torch.multiprocessing import Process
import torch.nn.functional as F
import time

class Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(Layer, self).__init__()
        self.layer = nn.Linear(input_size, output_size)
    def forward(self, x):
        x = F.relu(self.layer(x))
        return x




"""
def run(layer, size):
    if dist.get_rank() == 0:
        tensor = torch.randn(10)
        result = layer(tensor)
        print(result)
        send = dist.isend(tensor=result, dst=1)
        rec_space = torch.zeros(2)
        rec = dist.irecv(tensor=rec_space, src=1)
        if rec.is_completed():
            print("recive from 1:")
            print(rec_space)

    elif dist.get_rank() == 1:
        rec_tensor = torch.zeros(4)
        rec = dist.irecv(tensor=rec_tensor, src=0)
        print('Rank 1 recv....')
        print(rec_tensor)
        rank1result = layer(rec_tensor)
        print(rank1result)
        rank1send = dist.isend(tensor=rank1result, dst=0)
"""
class Flag(object):
    def __init__(self):
        self.flag = False
    def is_stop(self):
        return self.flag
    def set_stop(self):
        self.flag = True

def run(layer, size, f):
    print("init-" + str(dist.get_rank()))

    if dist.get_rank() == 0:
        send_opt = dist.isend(torch.randn(1), dst=1)
        send_opt.wait()
        if send_opt.is_completed():
            print("send_opt iscompleted")
        else:
            print("send_opt not completed")

    elif dist.get_rank() == 1:
        rec_opt = dist.irecv(torch.randn(1),src=0)
        rec_opt.wait()
        print("okokok")



def init_processes(fn, path, size):
    print("init_process......")
    dist.init_process_group(backend='tcp',init_method=path, world_size=size)
    if dist.get_rank() == 0:
        input_size = 10
        output_size = 4
    elif dist.get_rank() == 1:
        input_size = 4
        output_size = 2
    layer = Layer(input_size=input_size, output_size=output_size)
    f = Flag()
    fn(layer, size, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-size', type=int, help='input the sum of node')
    parser.add_argument('-path', help='the path fo share file system')

    args = parser.parse_args()
    processes = []
    print("size:" + str(args.size))
    print("path:" + args.path)
    for rank in range(args.size):
        p = Process(target=init_processes, args=(run, args.path, args.size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
