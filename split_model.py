import torch
from torch.autograd import Variable as V
import torch.distributed as dist
from torch import nn as nn
import argparse
from torch.multiprocessing import Process
import torch.nn.functional as F
import torch.optim as optim


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


def run(layer, size, batcher):
    print("init-"+ str(dist.get_rank()))
    iter = 0
    if dist.get_rank() == 0:
        start_opt = dist.isend(torch.randn(1), dst=1)
        start_opt.wait()
        while batcher.has_stop() == False:
            fro_send_opt = None
            fro_send_val = None
            back_rec_val = torch.randn(16)
            back_rec_opt = dist.irecv(tensor=back_rec_val,src=1)
            count = 0
            print("rank 0 start........")
            while count < 5 and not back_rec_opt.is_completed():
                #init_val= torch.randn(20)
                init_val = batcher.batch()
                fro_send_val = layer(init_val)
                print("rank 0 " + str(count))
                fro_send_opt = dist.isend(tensor = fro_send_val, dst=1)
                count += 1
            count = 0
            if back_rec_opt.is_completed():
                print("rank 0 rev" + str(iter))
                optimizer = optim.SGD(layer.parameters())
                optimizer.zero_grad()
                for_send_val.backward(back_rec_val)
                optimizer.step()
                iter += 1
                if iter > 10:
                    batcher.stop()
            else:
                send_opt.wait()
    if dist.get_rank() == 1:
        rec_opt = dist.irecv(torch.randn(1), src=0)
        rec_opt.wait()
        start_opt = dist.isend(torch.randn(1), dst=2)
        start_opt.wait()
        while batcher.has_stop() == False:
            fro_send_opt = None
            back_send_opt = None
            back_rec_val = torch.randn(10)
            back_rec_opt = dist.irecv(tensor=back_rec_val,src=2)
            fro_rec_val = torch.randn(16)
            fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=0)
            

            print("rank 1 start.......")
            if back_rec_opt.is_completed():
                print("rank 1 back rec")
                fro_rec_val.requires_grad = True
                optimizer = optim.SGD([fro_rec_val, layer.parameters()])
                optimizer.zero_grad()
                fro_send_val.backward(back_rec_val)
                optimizer.step()
                back_send_val = fro_rec_val.grad
                back_send_opt = dist.isend(tensor=back_send_val, src=0)
            else:
                count = 0
                while count < 5 and fro_rec_opt.is_completed():
                    print("rank 1 fro_rec")
                    fro_send_val  = layer(fro_rec_val)
                    fro_send_opt = dist.isend(tensor=fro_send_val, dis=2)
                    count += 1
                    fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=0)
    if dist.get_rank() == 2:
        rec_opt = dist.irecv(torch.randn(1), src=1)
        rec_opt.wait()
        start_opt = dist.isend(torch.randn(1), dst=3)
        start_opt.wait()
        while batcher.has_stop() == False:
            fro_send_opt = None
            back_send_opt = None
            back_rec_val = torch.randn(6)
            back_rec_opt = dist.irecv(tensor=back_rec_val,src=3)
            fro_rec_val = torch.randn(10)
            fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=1)
            
            print("rank 2 start..........")
            if back_rec_opt.is_completed():
                print("rank 2 back rec")
                fro_rec_val.requires_grad = True
                optimizer = optim.SGD([fro_rec_val, layer.parameters()])
                optimizer.zero_grad()
                fro_send_val.backward(back_rec_val)
                optimizer.step()
                back_send_val = fro_rec_val.grad
                back_send_opt = dist.isend(tensor=back_send_val, src=1)
            else:
                count = 0
                while count < 5 and fro_rec_opt.is_completed():
                    print("rank2 fro rec")
                    fro_send_val  = layer(fro_rec_val)
                    fro_send_opt = dist.isend(tensor=fro_send_val, dis=3)
                    count += 1
                    fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=1)
    if dist.get_rank() == 3:
        rec_opt = dist.irecv(torch.randn(1), src=2)
        rec_opt.wait()
        start_opt = dist.isend(torch.randn(1), dst=4)
        start_opt.wait()
        while batcher.has_stop() == False:
            fro_send_opt = None
            back_send_opt = None
            back_rec_val = torch.randn(2)
            back_rec_opt = dist.irecv(tensor=back_rec_val,src=4)
            fro_rec_val = torch.randn(6)
            fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=2)
            print("rank 3 start......")
            if back_rec_opt.is_completed():
                print("rank 3 back rec")
                fro_rec_val.requires_grad = True
                optimizer = optim.SGD([fro_rec_val, layer.parameters()])
                optimizer.zero_grad()
                fro_send_val.backward(back_rec_val)
                optimizer.step()
                back_send_val = fro_rec_val.grad
                back_send_opt = dist.isend(tensor=back_send_val, dis=2)
            else:
                count = 0
                while count < 5 and fro_rec_opt.is_completed():
                    print("rank 3 fro rec")
                    fro_send_val  = layer(fro_rec_val)
                    fro_send_opt = dist.isend(tensor=fro_send_val, dis=4)
                    count += 1
                    fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=2)
    if dist.get_rank() == 4:

        rec_opt = dist.irecv(torch.randn(1), src=3)
        rec_opt.wait()
        while batcher.has_stop() == False:
            print("rank 4 start......")
            fro_rec_val = torch.randn(4)
            fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=3)
            if fro_rec_opt.is_completed():
                print("rank 4 fro rec")
                fro_rec_val.requires_grad = True
                output = layer(fro_rec_val)
                optimizer = optim.SGD([fro_rec_val, layer.parameters()])
                optimizer.zero_grad()
                #target = torch.rand(1)
                target = batcher.get_target()
                criterion = nn.MSELoss()
                loss = criterion(output, target)
                print(str(iter) + "-loss: " + str(loss))
                loss.backward()
                optimizer.step()
                back_send_val = fro_rec_val.grad
                back_send_opt = dist.isend(tensor=back_send_val, dis=3)
    print("end.......")
                                                                                                                                                                                                                
        





def init_processes(fn, path, size, batcher):
    print("init_processs...")
    dist.init_process_group(backend='tcp',init_method=path, world_size=size)
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
