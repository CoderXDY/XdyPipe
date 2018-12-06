import torch
from torch.autograd import Variable as V
import torch.distributed as dist
from torch import nn as nn
import argparse
from torch.multiprocessing import Process
import torch.nn.functional as F
import torch.optim as optim
import queue


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
    print("init-" + str(dist.get_rank()))
    iter = 0
    q_input = queue.Queue()
    q_output = queue.Queue()
    if dist.get_rank() == 0:
        while iter < 10: 
            back_rec_val = torch.randn(16)
            back_rec_opt = dist.irecv(tensor=back_rec_val, src=1)
            count = 0
            print("rank0-iter:" + "-" + str(iter))
            while not back_rec_opt.is_completed():
                if count < 5:
                    init_val = batcher.batch()
                    fro_send_val = layer(init_val)
                    print("rank0 fro_send....")
                    #fro_send_opt = dist.isend(tensor=fro_send_val, dst=1)
                    dist.send(tensor=fro_send_val, dst=1)
                    count += 1
                else:
                    print("rank0 wait")
                    back_rec_opt.wait()
                    optimizer = optim.SGD(layer.parameters())
                    optimizer.zero_grad()
                    fro_send_val.backward(back_rec_val)
                    optimizer.step()
                    iter += 1
                    if iter >= 10:
                        break
                    else:
                        back_rec_val = torch.randn(16)
                        back_rec_opt = dist.irecv(tensor=back_rec_val, src=1)
                        count = 0
            else:
                optimizer = optim.SGD(layer.parameters())
                optimizer.zero_grad()
                for_send_val.backward(back_rec_val)
                optimizer.step()
                iter += 1


    if dist.get_rank() == 1:
        back_rec_val = torch.randn(10)
        back_rec_opt = dist.irecv(tensor=back_rec_val, src=2)
        fro_rec_val = torch.randn(16)
        fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=0)

        while fro_rec_opt.is_completed() or back_rec_opt.is_completed():
            print("rank1-iter:" + str(iter))
            if fro_rec_opt.is_completed():
                print("rank1 fro_rec...")
                q_input.put(fro_rec_val, block=False)
                fro_send_val = layer(fro_rec_val)
                #fro_send_opt = dist.isend(tensor=fro_send_val, dis=2)
                dist.send(tensor=fro_send_val, dis=2)
                q_output.put(fro_send_val, block=False)
                fro_rec_val = torch.randn(16)
                fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=0)
            elif back_rec_opt.is_completed():
                print("rank1 back_rec")
                fro_rec_val = q_input.get(block=False)
                fro_rec_val.requires_grad = True
                optimizer = optim.SGD([fro_rec_val, layer.parameters()])
                optimizer.zero_grad()
                fro_send_val = q_output.get(block=False)
                fro_send_val.backward(back_rec_val)
                optimizer.step()
                back_send_val = fro_rec_val.grad
                #back_send_opt = dist.isend(tensor=back_send_val, src=0)
                dist.send(tensor=back_send_val, src=0)
                iter += 1
                if iter >= 10:
                    break
                else:
                    back_rec_val = torch.randn(10)
                    back_rec_opt = dist.irecv(tensor=back_rec_val, src=2)




    if dist.get_rank() == 2:
        back_rec_val = torch.randn(6)
        back_rec_opt = dist.irecv(tensor=back_rec_val, src=3)
        fro_rec_val = torch.randn(10)
        fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=1)
        while fro_rec_opt.is_completed() or back_rec_opt.is_completed():
            print("rank2-iter" + str(iter))
            if fro_rec_opt.is_completed():
                print("rank2 fro_rec")
                q_input.put(fro_rec_val, block=False)
                fro_send_val = layer(fro_rec_val)
                #fro_send_opt = dist.isend(tensor=fro_send_val, dis=3)
                dist.send(tensor=fro_send_val, dis=3)
                q_output.put(fro_send_val, block=False)
                fro_rec_val = torch.randn(10)
                fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=1)
            elif back_rec_opt.is_completed():
                print("rank2 back_rec")
                fro_rec_val = q_input.get(block=False)
                fro_rec_val.requires_grad = True
                optimizer = optim.SGD([fro_rec_val, layer.parameters()])
                optimizer.zero_grad()
                fro_send_val = q_output.get(block=False)
                fro_send_val.backward(back_rec_val)
                optimizer.step()
                back_send_val = fro_rec_val.grad
                #back_send_opt = dist.isend(tensor=back_send_val, src=1)
                dist.send(tensor=back_send_val, src=1)
                iter += 1
                if iter >= 10:
                    break
                else:
                    back_rec_val = torch.randn(6)
                    back_rec_opt = dist.irecv(tensor=back_rec_val, src=3)



    if dist.get_rank() == 3:
        back_rec_val = torch.randn(2)
        back_rec_opt = dist.irecv(tensor=back_rec_val, src=4)
        fro_rec_val = torch.randn(6)
        fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=2)
        while fro_rec_opt.is_completed() or back_rec_opt.is_completed():
            print("rank3-iter:" +str(iter))
            if fro_rec_opt.is_completed():
                print("rank3 fro_rec")
                q_input.put(fro_rec_val, block=False)
                fro_send_val = layer(fro_rec_val)
                #fro_send_opt = dist.isend(tensor=fro_send_val, dis=4)
                dist.send(tensor=fro_send_val, dis=4)
                q_output.put(fro_send_val, block=False)
                fro_rec_val = torch.randn(6)
                fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=2)
            elif back_rec_opt.is_completed():
                print("rank3 back_rec")
                fro_rec_val = q_input.get(block=False)
                fro_rec_val.requires_grad = True
                optimizer = optim.SGD([fro_rec_val, layer.parameters()])
                optimizer.zero_grad()
                fro_send_val = q_output.get(block=False)
                fro_send_val.backward(back_rec_val)
                optimizer.step()
                back_send_val = fro_rec_val.grad
                #back_send_opt = dist.isend(tensor=back_send_val, src=2)
                dist.send(tensor=back_send_val, src=2)
                iter += 1
                if iter >= 10:
                    break
                else:
                    back_rec_val = torch.randn(2)
                    back_rec_opt = dist.irecv(tensor=back_rec_val, src=4)



    if dist.get_rank() == 4:

        fro_rec_val = torch.randn(2)
        fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=3)

        while fro_rec_opt.is_completed():
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
            #back_send_opt = dist.isend(tensor=back_send_val, dis=3)
            dist.send(tensor=back_send_val, dis=3)
            iter += 1
            if iter >= 10:
                break
            else:
                fro_rec_val = torch.randn(2)
                fro_rec_opt = dist.irecv(tensor=fro_rec_val, src=3)







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
