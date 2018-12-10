import torch
from torch.autograd import Variable as V
import torch.distributed as dist
from torch import nn as nn
import argparse
from torch.multiprocessing import Process, Queue, Value
import torch.nn.functional as F
import torch.optim as optim
import logging as Log
import time

"""



toy_expand_pipeline.py -------(precomputing strategy)------- expend_precomputing_pipeline.py

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




def run(qs, layer):

    if dist.get_rank() == 0:
        input_x = torch.ones(20, requires_grad=True)
        output_x = layer(input_x)
        input_x.share_memory_()
        qs[0].put(input_x)
        send_opt = dist.isend(tensor=output_x, dst=1)
        send_opt.wait()
        time.sleep(20)
        print("===================rank 0 share=================================")
        for name, param in layer.named_parameters():
            print(name + "->")
            print(param)
    elif dist.get_rank() == 1:

        rec_val = torch.zeros(10, requires_grad=True)
        dist.recv(tensor=rec_val, src=0)
        rec_val.share_memory_()
        qs[1].put(rec_val)
        send_opt = dist.isend(tensor=torch.ones(1), dst=2)
        send_opt.wait()
        time.sleep(20)
        print("===================rank 1 share=================================")
        for name, param in layer.named_parameters():
            print(name + "->")
            print(param)
    elif dist.get_rank() == 2:
        rec_val = torch.randn(1)
        dist.recv(tensor=rec_val, src=1)
        print("===================" + str(qs[1].qsize()))
        input_x = qs[1].get()
        input_x.requires_grad_()
        print("input_x:")
        print(input_x)
        print("before layer parmeters")
        for name, param in layer.named_parameters():
            print(name + "->")
            print(param)
        output_x = layer(input_x)
        optimizer = optim.SGD(layer.parameters(), lr=0.01)
        optimizer.zero_grad()
        criterion = nn.MSELoss()
        target_v = torch.randn(1)
        loss = criterion(output_x, target_v)
        loss.backward()

        optimizer.step()
        print(loss.grad)
        print(loss.grad_fn)
        print(loss.requires_grad)
        print("-loss: " + str(loss))

        print("-------------------------------------")
        print(input_x.grad)
        print(input_x.requires_grad)
        print("after layer poarmeters")
        for name, param in layer.named_parameters():
            print(name + "->")
            print(param)
        send_opt = dist.isend(tensor=input_x.grad, dst=3)
        send_opt.wait()
        time.sleep(10)
    elif dist.get_rank() == 3:
        back_grad = torch.zeros(10, requires_grad=True)
        dist.recv(tensor=back_grad, src=2)
        print("back_grad")
        print(back_grad)
        input_x = qs[0].get()
        input_x.requires_grad_()
        print("rank3 before...........")
        for name, param in layer.named_parameters():
            print(name + "->")
            print(param)
        output_x = layer(input_x)
        optimizer = optim.SGD(layer.parameters(), lr=0.01)
        optimizer.zero_grad()
        output_x.backward(back_grad)
        optimizer.step()
        print("rank3 after...........")
        for name, param in layer.named_parameters():
            print(name + "->")
            print(param)
        print("rank3")
        print(input_x.grad)
        time.sleep(10)



def init_processes(fn, path, size, q, layers):
    dist.init_process_group(backend='tcp', init_method=path, world_size=size, rank=rank)
    if dist.get_rank() == 0 or dist.get_rank() == 3:
        fn(q, layers[0])
    elif dist.get_rank() == 1 or dist.get_rank() == 2:
        fn(q, layers[1])



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-size', type=int, help='input the sum of node')
    parser.add_argument('-path', help='the path fo share file system')
    args = parser.parse_args()
    processes = []
    print("size:" + str(args.size))
    print("path:" + args.path)
    q = Queue(100)
    q1 = Queue(100)
    qs = []
    qs.append(q)
    qs.append(q1)

    layers = []

    layer03 = Layer(20, 10)
    layer03.share_memory()
    layers.append(layer03)

    layer12 = Layer(10, 1)
    layer12.share_memory()
    layers.append(layer12)


    for rank in range(args.size):
        p = Process(target=init_processes, args=(run, args.path, args.size, qs, layers))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
