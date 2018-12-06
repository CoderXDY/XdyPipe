import torch
from torch.autograd import Variable as V
import torch.distributed as dist
from torch import nn as nn
import argparse
from torch.multiprocessing import Process
import torch.nn.functional as F
import torch.optim as optim
import queue
import logging as log

log.basicConfig(filename='pipeline2.log', level=log.INFO)

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
    q_input = queue.Queue()
    q_output = queue.Queue()
    count = 0
    iter = 0
    try:
        for i in range(5):
            if dist.get_rank() == 0:
                back_rec_val = torch.randn(16)
                back_rec_opt = dist.irecv(tensor=back_rec_val, src=1)
                if not back_rec_opt.is_completed():
                    print("rank 0 not back rec")
                    fro_send_val = torch.randn(16)
                    fro_send_opt = dist.isend(tensor=fro_send_val, dst=1)
                    fro_send_opt.wait()
                    print("rank 0 after wait....")
                else:
                    print("rank 0 back rec")

            elif dist.get_rank() == 1:
                back_rec_val = torch.randn(10)
                back_rec_opt = dist.irecv(tensor=back_rec_val, src=2)

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
                else:
                    fro_rec_val = torch.randn(16)
                    fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=0)
                    if fro_rec_opt.is_completed():
                        pass
                    else:
                        pass

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



            elif dist.get_rank() == 2:
                print("rank 2")
            elif dist.get_rank() == 3:
                print("rank 3")
            elif dist.get_rank() == 4:
                print("rank 4")



    except Exception as e:
        print(e)














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
