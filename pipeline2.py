import torch
from torch.autograd import Variable as V
import torch.distributed as dist
from torch import nn as nn
import argparse
from torch.multiprocessing import Process
import torch.nn.functional as F
import torch.optim as optim
import queue
import logging as Log

Log.basicConfig(filename='pipeline2.log', level=Log.INFO)


class Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(Layer, self).__init__()
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer(x))
        return x


class Batcher(object):
    def __init__(self):
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
"""
def run(layer, size, batcher):
    print("init-" + str(dist.get_rank()))
    iter = 0
    q_input = queue.Queue()
    q_output = queue.Queue()
    count = 0
    try:
        while not batcher.has_stop():
            if dist.get_rank() == 0:
                print("rank" + str(dist.get_rank()) + "-iter:" + str(iter))
                back_rec_val = torch.randn(16)
                back_rec_opt = dist.irecv(tensor=back_rec_val, src=1)
                if not back_rec_opt.is_completed():
                    if count < 5:
                        init_val = batcher.batch()
                        fro_send_val = layer(init_val)
                        q_output.put(fro_send_val, block=False)
                        print("rank0 fro_send....")
                        fro_send_opt = dist.isend(tensor=fro_send_val, dst=1)
                        count += 1
                        fro_send_opt.wait()
                    else:
                        back_rec_opt.wait()
                else:
                    optimizer = optim.SGD(layer.parameters())
                    optimizer.zero_grad()
                    fro_send_val = q_output.get(block=False)
                    fro_send_val.backward(back_rec_val)
                    optimizer.step()
                    count -= 1
                    iter += 1
                    if iter > 10:
                        batcher.stop()
            if dist.get_rank() == 1:
                print("rank" + str(dist.get_rank()) + "-iter:" + str(iter))
                back_rec_val = torch.randn(10)
                back_rec_opt = dist.irecv(tensor=back_rec_val, src=2)
                fro_rec_val = torch.randn(16)
                fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=0)
                if back_rec_opt.is_completed():
                    fro_rec_val = q_input.get(block=False)
                    fro_rec_val.requires_grad = True
                    optimizer = optim.SGD([fro_rec_val, layer.parameters()])
                    optimizer.zero_grad()
                    fro_send_val = q_output.get(block=False)
                    fro_send_val.backward(back_rec_val)
                    optimizer.step()
                    back_send_val = fro_rec_val.grad
                    back_send_opt = dist.isend(tensor=back_send_val, src=0)
                    back_send_opt.wait()
                elif fro_rec_opt.is_completed():
                    print("rank1 fro_rec...")
                    q_input.put(fro_rec_val, block=False)
                    fro_send_val = layer(fro_rec_val)
                    q_output.put(fro_send_val, block=False)
                    fro_send_opt = dist.isend(tensor=fro_send_val, dis=2)
                    fro_send_opt.wait()
            if dist.get_rank() == 2:
                print("rank" + str(dist.get_rank()) + "-iter:" + str(iter))
                back_rec_val = torch.randn(6)
                back_rec_opt = dist.irecv(tensor=back_rec_val, src=3)
                fro_rec_val = torch.randn(10)
                fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=1)
                if back_rec_opt.is_completed():
                    print("rank2 back_rec")
                    fro_rec_val = q_input.get(block=False)
                    fro_rec_val.requires_grad = True
                    optimizer = optim.SGD([fro_rec_val, layer.parameters()])
                    optimizer.zero_grad()
                    fro_send_val = q_output.get(block=False)
                    fro_send_val.backward(back_rec_val)
                    optimizer.step()
                    back_send_val = fro_rec_val.grad
                    back_send_opt = dist.isend(tensor=back_send_val, src=1)
                    back_send_opt.wait()
                elif fro_rec_opt.is_completed():
                    print("rank2 fro_rec")
                    q_input.put(fro_rec_val, block=False)
                    fro_send_val = layer(fro_rec_val)
                    q_output.put(fro_send_val, block=False)
                    fro_send_opt = dist.isend(tensor=fro_send_val, dis=3)
                    fro_send_opt.wait()
            if dist.get_rank() == 3:
                print("rank" + str(dist.get_rank()) + "-iter:" + str(iter))
                back_rec_val = torch.randn(2)
                back_rec_opt = dist.irecv(tensor=back_rec_val, src=4)
                fro_rec_val = torch.randn(6)
                fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=2)
                if back_rec_opt.is_completed():
                    print("rank3 back_rec")
                    fro_rec_val = q_input.get(block=False)
                    fro_rec_val.requires_grad = True
                    optimizer = optim.SGD([fro_rec_val, layer.parameters()])
                    optimizer.zero_grad()
                    fro_send_val = q_output.get(block=False)
                    fro_send_val.backward(back_rec_val)
                    optimizer.step()
                    back_send_val = fro_rec_val.grad
                    back_send_opt = dist.isend(tensor=back_send_val, src=2)
                    back_send_opt.wait()
                elif fro_rec_opt.is_completed():
                    print("rank3 fro_rec")
                    q_input.put(fro_rec_val, block=False)
                    fro_send_val = layer(fro_rec_val)
                    q_output.put(fro_send_val, block=False)
                    fro_send_opt = dist.isend(tensor=fro_send_val, dis=4)
                    fro_send_opt.wait()
            if dist.get_rank() == 4:
                print("rank" + str(dist.get_rank()) + "-iter:" + str(iter))
                fro_rec_val = torch.randn(2)
                fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=3)
                if fro_rec_opt.is_completed():
                    print("rank4-iter:" + str(iter))
                    fro_rec_val.requires_grad = True
                    output = layer(fro_rec_val)
                    optimizer = optim.SGD([fro_rec_val, layer.parameters()])
                    optimizer.zero_grad()
                    target = batcher.get_target()
                    criterion = nn.MSELoss()
                    loss = criterion(output, target)
                    print(str(iter) + "-loss: " + str(loss))
                    loss.backward()
                    optimizer.step()
                    back_send_val = fro_rec_val.grad
                    back_send_opt = dist.isend(tensor=back_send_val, dis=3)
                    back_send_opt.wait()
                else:
                    fro_rec_opt.wait()
    except Exception as e:
        print(e)
"""




def run(layer, size, batcher):
    print("init-" + str(dist.get_rank()))
    iter = 0
    q_input = queue.Queue()
    q_output = queue.Queue()
    count = 0
    try:
        if dist.get_rank() == 0:
            print("rank" + str(dist.get_rank()) + "-iter:" + str(iter))
            back_rec_val = torch.randn(16)
            back_rec_opt = dist.irecv(tensor=back_rec_val, src=1)
            while not batcher.has_stop():
                print("rank 0 batcher.has_stop()")

                if not back_rec_opt.is_completed():
                    print("rank 0 in back_rec_opt not completed")

                    if count < 5:
                        init_val = batcher.batch()
                        fro_send_val = layer(init_val)
                        q_output.put(fro_send_val, block=False)
                        print("rank0 fro_send....")
                        fro_send_opt = dist.isend(tensor=fro_send_val, dst=1)
                        count += 1
                        fro_send_opt.wait()
                    else:
                        print("rank 0 back_rec_opt.wait---------------------------")
                        back_rec_opt.wait()

                else:
                    print("rank 9 in back_rec_opt completed.....")
                    
                    optimizer = optim.SGD(layer.parameters())
                    optimizer.zero_grad()
                    fro_send_val = q_output.get(block=False)
                    fro_send_val.backward(back_rec_val)
                    optimizer.step()
                    count -= 1
                    iter += 1
                    if iter > 10:
                        batcher.stop()
                    else:
                        back_rec_val = torch.randn(16)
                        back_rec_opt = dist.irecv(tensor=back_rec_val, src=1)
                    
        if dist.get_rank() == 1:
            print("rank" + str(dist.get_rank()) + "-iter:" + str(iter))
            back_rec_val = torch.randn(10)
            back_rec_opt = dist.irecv(tensor=back_rec_val, src=2)
            fro_rec_val = torch.randn(16)
            fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=0)
            while not batcher.has_stop():
                print("rank 1 no stop")
                if back_rec_opt.is_completed():
                    fro_rec_val = q_input.get(block=False)
                    fro_rec_val.requires_grad = True
                    optimizer = optim.SGD([fro_rec_val, layer.parameters()])
                    optimizer.zero_grad()
                    fro_send_val = q_output.get(block=False)
                    fro_send_val.backward(back_rec_val)
                    optimizer.step()
                    back_rec_val = torch.randn(10)
                    back_rec_opt = dist.irecv(tensor=back_rec_val, src=2)
                    back_send_val = fro_rec_val.grad
                    back_send_opt = dist.isend(tensor=back_send_val, src=0)
                    back_send_opt.wait()
                elif fro_rec_opt.is_completed():
                    print("rank1 fro_rec...")
                    q_input.put(fro_rec_val, block=False)
                    fro_send_val = layer(fro_rec_val)
                    q_output.put(fro_send_val, block=False)
                    fro_send_opt = dist.isend(tensor=fro_send_val, dis=2)
                    fro_rec_val = torch.randn(16)
                    fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=0)
                    fro_send_opt.wait()
                else:
                    if count < 4:
                        print("rank 1 count < 4 ")
                        fro_rec_opt.wait()
                    else:
                        print("rank 1 count > 4")
                        back_rec_opt.wait()



        if dist.get_rank() == 2:
            print("rank" + str(dist.get_rank()) + "-iter:" + str(iter))
            back_rec_val = torch.randn(6)
            back_rec_opt = dist.irecv(tensor=back_rec_val, src=3)
            fro_rec_val = torch.randn(10)
            fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=1)
            while not batcher.has_stop():
                if back_rec_opt.is_completed():
                    print("rank2 back_rec")
                    fro_rec_val = q_input.get(block=False)
                    fro_rec_val.requires_grad = True
                    optimizer = optim.SGD([fro_rec_val, layer.parameters()])
                    optimizer.zero_grad()
                    fro_send_val = q_output.get(block=False)
                    fro_send_val.backward(back_rec_val)
                    optimizer.step()
                    back_rec_val = torch.randn(6)
                    back_rec_opt = dist.irecv(tensor=back_rec_val, src=3)
                    back_send_val = fro_rec_val.grad
                    back_send_opt = dist.isend(tensor=back_send_val, src=1)
                    back_send_opt.wait()
                elif fro_rec_opt.is_completed():
                    print("rank2 fro_rec")
                    q_input.put(fro_rec_val, block=False)
                    fro_send_val = layer(fro_rec_val)
                    q_output.put(fro_send_val, block=False)
                    fro_send_opt = dist.isend(tensor=fro_send_val, dis=3)
                    fro_rec_val = torch.randn(10)
                    fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=1)
                    fro_send_opt.wait()
                else:
                    if count < 4:
                        fro_rec_opt.wait()
                    else:
                        back_rec_opt.wait()


        if dist.get_rank() == 3:
            print("rank" + str(dist.get_rank()) + "-iter:" + str(iter))
            back_rec_val = torch.randn(2)
            back_rec_opt = dist.irecv(tensor=back_rec_val, src=4)
            fro_rec_val = torch.randn(6)
            fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=2)
            while not batcher.has_stop():
                if back_rec_opt.is_completed():
                    print("rank3 back_rec")
                    fro_rec_val = q_input.get(block=False)
                    fro_rec_val.requires_grad = True
                    optimizer = optim.SGD([fro_rec_val, layer.parameters()])
                    optimizer.zero_grad()
                    fro_send_val = q_output.get(block=False)
                    fro_send_val.backward(back_rec_val)
                    optimizer.step()
                    back_rec_val = torch.randn(2)
                    back_rec_opt = dist.irecv(tensor=back_rec_val, src=4)
                    back_send_val = fro_rec_val.grad
                    back_send_opt = dist.isend(tensor=back_send_val, src=2)
                    back_send_opt.wait()
                elif fro_rec_opt.is_completed():
                    print("rank3 fro_rec")
                    q_input.put(fro_rec_val, block=False)
                    fro_send_val = layer(fro_rec_val)
                    q_output.put(fro_send_val, block=False)
                    fro_send_opt = dist.isend(tensor=fro_send_val, dis=4)
                    fro_rec_val = torch.randn(6)
                    fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=2)
                    fro_send_opt.wait()
                else:
                    if count < 4:
                        fro_rec_opt.wait()
                    else:
                        back_rec_opt.wait()


        if dist.get_rank() == 4:
            print("rank" + str(dist.get_rank()) + "-iter:" + str(iter))
            fro_rec_val = torch.randn(2)
            fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=3)
            while not batcher.has_stop():
                if fro_rec_opt.is_completed():
                    print("rank4-iter:" + str(iter))
                    fro_rec_val.requires_grad = True
                    output = layer(fro_rec_val)
                    optimizer = optim.SGD([fro_rec_val, layer.parameters()])
                    optimizer.zero_grad()
                    target = batcher.get_target()
                    criterion = nn.MSELoss()
                    loss = criterion(output, target)
                    print(str(iter) + "-loss: " + str(loss))
                    loss.backward()
                    optimizer.step()
                    back_send_val = fro_rec_val.grad
                    back_send_opt = dist.isend(tensor=back_send_val, dis=3)
                    fro_rec_val = torch.randn(2)
                    fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=3)
                    back_send_opt.wait()
                else:
                    fro_rec_opt.wait()


    except Exception as e:
        Log.error(e)










def init_processes(fn, path, size, batcher):

    dist.init_process_group(backend='tcp', init_method=path, world_size=size)
    if dist.get_rank() == 0:
        input_size = 20
        output_size = 16
    elif dist.get_rank() == 1:
        input_size = 16
        output_size = 10
    elif dist.get_rank() == 2:
        input_size = 10
        output_size = 6
    elif dist.get_rank() == 3:
        input_size = 6
        output_size = 2
    elif dist.get_rank() == 4:
        input_size = 2
        output_size = 1
    layer = Layer(input_size=input_size, output_size=output_size)
    fn(layer, size, batcher)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-size', type=int, help='input the sum of node')
    parser.add_argument('-path', help='the path fo share file system')
    args = parser.parse_args()
    processes = []
    print("size:" + str(args.size))
    print("path:" + args.path)
    batcher = Batcher()
    for rank in range(args.size):
        p = Process(target=init_processes, args=(run, args.path, args.size, batcher))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
