import torch
from torch.autograd import Variable as V
import torch.distributed as dist
from torch import nn as nn
import argparse
from torch.multiprocessing import Process, Queue
import torch.nn.functional as F
import torch.optim as optim
import logging as Log

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



def run(layer, size, batcher):
    q0 = Queue()
    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    q4 = Queue()
    for iter in range(10):
        print("rank" + str(dist.get_rank()) + "-" + str(iter) + " start.....")
        if dist.get_rank() == 0:
            input_data = batcher.batch()
            input_v = V(input_data)
            q0.put(input_v, block=False)
            output_v = layer(input_v)
            send_val = output_v.data
            send_opt = dist.isend(tensor=send_val, dst=1)
            send_opt.wait()
            print("rank" + str(dist.get_rank()) + " is send.....")
        elif dist.get_rank() == 1:
            rec_val = torch.randn(16)
            dist.recv(tensor=rec_val, src=0)
            input_v = V(rec_val)
            q1.put(input_v, block=False)
            output_v = layer(input_v)
            send_val = output_v.data
            send_opt = dist.isend(tensor=send_val, dst=2)
            send_opt.wait()
            print("rank" + str(dist.get_rank()) + " is send.....")
        elif dist.get_rank() == 2:
            rec_val = torch.randn(10)
            dist.recv(tensor=rec_val, src=1)
            input_v = V(rec_val)
            q2.put(input_v, block=False)
            output_v = layer(input_v)
            send_val = output_v.data
            send_opt = dist.isend(tensor=send_val, dst=3)
            send_opt.wait()
            print("rank" + str(dist.get_rank()) + " is send.....")
        elif dist.get_rank() == 3:
            rec_val = torch.randn(6)
            dist.recv(tensor=rec_val, src=2)
            input_v = V(rec_val)
            q3.put(input_v, block=False)
            output_v = layer(input_v)
            send_val = output_v.data
            send_opt = dist.isend(tensor=send_val, dst=4)
            send_opt.wait()
            print("rank" + str(dist.get_rank()) + " is send.....")
        elif dist.get_rank() == 4:
            rec_val = torch.rand(2)
            dist.recv(tensor=rec_val, src=3)
            input_v = V(rec_val)
            q4.put(input_v, block=False)
            #output_v = layer(input_v)
            #send_val = output_v.data
            #send_opt = dist.isend(tensor=send_val, dst=5)
            #send_opt.wait()
            send_val = torch.rand(1)
            send_opt = dist.isend(tensor=send_val, dst=5)
            send_opt.wait()
            print("rank" + str(dist.get_rank()) + " is send.....")

        elif dist.get_rank() == 5:
            rec_val =torch.rand(1)
            dist.recv(tensor=rec_val, src=4)

            input_v = q4.get(block=False)
            output_v = layer(input_v)
            input_v.requires_grad = True
            optimizer = optim.SGD([input_v, layer.parameters()])
            optimizer.zero_grad()
            criterion = nn.MSELoss()
            target_data = batcher.get_target()
            target_v = V(target_data)
            loss = criterion(output_v, target_v)
            print(str(iter) + "-loss: " + str(loss))
            loss.backward()
            optimizer.step()

            send_val = input_v.grad.data
            send_opt = dist.isend(tensor=send_val, dst=6)
            send_opt.wait()
            print("rank" + str(dist.get_rank()) + " is send.....")
            
        elif dist.get_rank() == 6:
            back_grad = torch.randn(2)
            dist.recv(tensor=back_grad, src=5)

            input_v = q3.get(block=False)
            output_v = layer(input_v)
            input_v.requires_grad = True
            optimizer = optim.SGD([input_v, layer.parameters()])
            optimizer.zero_grad()
            output_v.backward(back_grad)
            optimizer.step()

            send_val = input_v.grad.data
            send_opt = dist.isend(tensor=send_val, dst=6)
            send_opt.wait()
            print("rank" + str(dist.get_rank()) + " is send.....")


        elif dist.get_rank() == 7:
            back_grad = torch.randn(6)
            dist.recv(tensor=back_grad, src=6)

            input_v = q2.get(block=False)
            output_v = layer(input_v)
            input_v.requires_grad = True
            optimizer = optim.SGD([input_v, layer.parameters()])
            optimizer.zero_grad()
            output_v.backward(back_grad)
            optimizer.step()

            send_val = input_v.grad.data
            send_opt = dist.isend(tensor=send_val, dst=7)
            send_opt.wait()
            print("rank" + str(dist.get_rank()) + " is send.....")
        elif dist.get_rank() == 8:
            back_grad = torch.randn(16)
            dist.recv(tensor=back_grad, src=7)

            input_v = q2.get(block=False)
            output_v = layer(input_v)
            input_v.requires_grad = True
            optimizer = optim.SGD([input_v, layer.parameters()])
            optimizer.zero_grad()
            output_v.backward(back_grad)
            optimizer.step()

            send_val = input_v.grad.data
            send_opt = dist.isend(tensor=send_val, dst=9)
            send_opt.wait()
            print("rank" + str(dist.get_rank()) + " is send.....")
        elif dist.get_rank() == 9:
            back_grad = torch.randn(20)
            dist.recv(tensor=back_grad, src=8)

            input_v = q0.get(block=False)
            output_v = layer(input_v)
            input_v.requires_grad = True
            optimizer = optim.SGD([input_v, layer.parameters()])
            optimizer.zero_grad()
            output_v.backward(back_grad)
            optimizer.step()
            print("rank" + str(dist.get_rank()) + " is send.....")








def init_processes(fn, path, size, batcher,layers):
    dist.init_process_group(backend='tcp', init_method=path, world_size=size)
    print("init-process-" + str(dist.get_rank()) + "......")
    if dist.get_rank() == 0 or dist.get_rank() == 9:
        fn(layers[0], size, batcher)
    elif dist.get_rank() == 1 or dist.get_rank() == 8:
        fn(layers[1], size, batcher)
    elif dist.get_rank() == 2 or dist.get_rank() == 7:
        fn(layers[2], size, batcher)
    elif dist.get_rank() == 3 or dist.get_rank() == 6:
        fn(layers[3], size, batcher)
    elif dist.get_rank() == 4 or dist.get_rank() == 5:
        fn(layers[4], size, batcher)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-size', type=int, help='input the sum of node')
    parser.add_argument('-path', help='the path fo share file system')
    args = parser.parse_args()
    processes = []
    print("size:" + str(args.size))
    print("path:" + args.path)
    
    batcher = Batcher()#todo

    layers = []
    
    layer09 = Layer(20, 16)
    layer09.share_memory()
    layers.append(layer09)
    
    layer18 = Layer(16, 10)
    layer18.share_memory()
    layers.append(layer18)
    
    layer27 = Layer(10, 6)
    layer27.share_memory()
    layers.append(layer27)
    
    layer36 = Layer(6, 2)
    layer36.share_memory()
    layers.append(layer36)
    
    layer45 = Layer(2, 1)
    layer45.share_memory()
    layers.append(layer45)
    
    for rank in range(args.size):
        p = Process(target=init_processes, args=(run, args.path, args.size, batcher, layers))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
