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

Log.basicConfig(filename='pipeline2.log', level=Log.INFO)

torch.manual_seed(10)


class Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(Layer, self).__init__()
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer(x))
        return x


class Batcher(object):
    def __init__(self):
        torch.manual_seed(10)
        self.data = list()
        self.index = 0
        self.target_index = 0
        self.is_stop = False
        for index in range(1100):
            x = torch.randn(20)
            y = torch.randn(1)
            self.data.append((x, y))

    def batch(self):
        result = self.data[self.index]
        self.index += 1
        return result[0]

    def get_target(self):
        result = self.data[self.target_index]
        self.target_index += 1
        return result[1]

    def has_stop(self):
        return self.is_stop

    def stop(self):
        self.is_stop = True


def run(layer, size, batcher, stop_flag):
    q0 = Queue(100)
    q1 = Queue(100)
    iter = 0
    try:
        while stop_flag.value == 0:
            print("rank" + str(dist.get_rank()) + "-" + str(iter) + " start.....")
            if dist.get_rank() == 0:
                input_data = batcher.batch()
                input_v = V(input_data)
                output_v = layer(input_v)
                send_val = output_v.data
                send_opt = dist.isend(tensor=send_val, dst=1)
                send_opt.wait()
                print("rank" + str(dist.get_rank()) + " is send.....")
            elif dist.get_rank() == 1:
                rec_val = torch.randn(10)
                dist.recv(tensor=rec_val, src=0)
                send_val = torch.randn(1)
                send_opt = dist.isend(tensor=send_val, dst=2)
                send_opt.wait()
                print("rank" + str(dist.get_rank()) + " is send.....")
            elif dist.get_rank() == 2:
                rec_val = torch.randn(1)
                dist.recv(tensor=rec_val, src=1)
                
                input_v = V(torch.randn(10))
                input_v.requires_grad = True
                output_v = layer(input_v)
                optimizer = optim.SGD(layer.parameters(), lr=0.01)
                optimizer.zero_grad()
                criterion = nn.MSELoss()
                target_data = batcher.get_target()
                target_v = V(target_data)
                loss = criterion(output_v, target_v)
                print(str(iter) + "-loss: " + str(loss))
                loss.backward()
                optimizer.step()

                send_val = input_v.grad.data
                print("size of send_val :" + str(send_val.size()))
                send_opt = dist.isend(tensor=send_val, dst=3)
                send_opt.wait()

                print("rank" + str(dist.get_rank()) + " is send.....")
            elif dist.get_rank() == 3:
                back_grad = torch.randn(10)
                dist.recv(tensor=back_grad, src=2)
                if iter > 100:
                    stop_flag.value = 1
                    print("rank" + str(dist.get_rank()) + "-iter" + str(iter) + " set stop......")
                else:
                    input_v = V(torch.randn(20))
                    input_v.requires_grad = True
                    output_v = layer(input_v)
                    optimizer = optim.SGD(layer.parameters(), lr=0.01)
                    optimizer.zero_grad()
                    output_v.backward(back_grad)
                    optimizer.step()
                    print("rank" + str(dist.get_rank()) + " is send.....")
            iter += 1

        print("rank" + str(dist.get_rank()) + ": stop.......")
    except Exception as e:
        print(e)
        return



def init_processes(fn, path, size, batcher, layers, stop_flag, rank):
    dist.init_process_group(backend='tcp', init_method=path, world_size=size, rank=rank)
    print(str(time.time()) + "  init-process-" + str(dist.get_rank()) + "......")
    if dist.get_rank() == 0 or dist.get_rank() == 3:
        fn(layers[0], size, batcher, stop_flag)
    elif dist.get_rank() == 1 or dist.get_rank() == 2:
        fn(layers[1], size, batcher, stop_flag)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-size', type=int, help='input the sum of node')
    parser.add_argument('-path', help='the path fo share file system')
    args = parser.parse_args()
    processes = []
    print("size:" + str(args.size))
    print("path:" + args.path)

    batcher = Batcher()  # todo

    layers = []

    layer03 = Layer(20, 10)
    layer03.share_memory()
    layers.append(layer03)

    layer12 = Layer(10, 1)
    layer12.share_memory()
    layers.append(layer12)

    stop_flag = Value('i', 0)

    for rank in range(args.size):
        p = Process(target=init_processes, args=(run, args.path, args.size, batcher, layers, stop_flag, rank))
        p.start()
        #time.sleep(2)
        processes.append(p)

    for p in processes:
        p.join()
