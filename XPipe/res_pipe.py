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
from res import BasicBlock, Bottleneck, ResInputLayer, ResBlockLayer, ResOutputLayer
import torchvision
import torchvision.transforms as transforms
from utils import progress_bar
import traceback




def run(queue, layer, stop_flag, loader=None, target_buffer=None):

    batch_size = 128
    batch_count = 0
    train_loss = 0
    try:
        while stop_flag.value == 0:

            if dist.get_rank() == 0:
                if loader is not None:
                    dataiter = iter(loader)
                    input_v, target_v = next(dataiter)
                    input_v.share_memory_()
                    queue.put(input_v)

                    target_v.share_memory_()
                    target_buffer.put(target_v)

                    output_v = layer(input_v)
                    send_opt = dist.isend(tensor=output_v, dst=1)
                    send_opt.wait()
                else:
                    raise Exception('loader error', "loader is None")
            elif dist.get_rank() == 1:
                rec_val = torch.zeros([batch_size, 64, 32, 32], requires_grad=True)
                dist.recv(tensor=rec_val, src=0)
                rec_val.share_memory_()
                queue.put(rec_val)

                output_v = layer(rec_val)

                send_opt = dist.isend(tensor=output_v, dst=2)
                send_opt.wait()

            elif dist.get_rank() == 2:
                rec_val = torch.zeros([batch_size, 64, 32, 32], requires_grad=True)
                dist.recv(tensor=rec_val, src=1)
                rec_val.share_memory_()
                queue.put(rec_val)

                output_v = layer(rec_val)

                send_opt = dist.isend(tensor=output_v, dst=3)
                send_opt.wait()
            elif dist.get_rank() == 3:
                rec_val = torch.zeros([batch_size, 128, 16, 16], requires_grad=True)
                dist.recv(tensor=rec_val, src=2)
                rec_val.share_memory_()
                queue.put(rec_val)

                output_v = layer(rec_val)
                send_opt = dist.isend(tensor=output_v, dst=4)
                send_opt.wait()
            elif dist.get_rank() == 4:
                rec_val = torch.zeros([batch_size, 256, 8, 8], requires_grad=True)
                dist.recv(tensor=rec_val, src=3)
                rec_val.share_memory_()
                queue.put(rec_val)

                output_v = layer(rec_val)

                send_opt = dist.isend(tensor=output_v, dst=5)
                send_opt.wait()
            elif dist.get_rank() == 5:
                rec_val = torch.zeros([batch_size, 512, 4, 4], requires_grad=True)
                dist.recv(tensor=rec_val, src=4)
                rec_val.share_memory_()
                queue.put(rec_val)

                send_opt = dist.isend(tensor=torch.randn(1), dst=6)
                send_opt.wait()
            elif dist.get_rank() == 6:
                rec_val = torch.zeros(1)
                dist.recv(tensor=rec_val, src=5)

                input_v = queue.get()
                input_v.requires_grad_()
                output_v = layer(input_v)

                optimizer = optim.SGD(layer.parameters(), lr=0.01)

                optimizer.zero_grad()

               # criterion = nn.MSELoss()
                criterion = nn.CrossEntropyLoss()
                target_v = target_buffer.get()
                #output_v = output_v.long()
                loss = criterion(output_v, target_v)
                loss.backward()

                optimizer.step()

                #train_loss += loss.item()
               # print("loss:" + str(loss.item()))
                #_, predicted = outputs.max(1)
                #total += targets.size(0)
                #correct += predicted.eq(targets).sum().item()

                #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #             % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

                send_opt = dist.isend(tensor=input_v.grad, dst=7)
                send_opt.wait()


            elif dist.get_rank() == 7:
                back_grad = torch.zeros([batch_size, 512, 4, 4], requires_grad=True)
                dist.recv(tensor=back_grad, src=6)
                input_v = queue.get()
                input_v.requires_grad_()
                output_v = layer(input_v)
     
                optimizer = optim.SGD(layer.parameters(), lr=0.01)
                optimizer.zero_grad()
                output_v.backward(back_grad)
                optimizer.step()
                send_opt = dist.isend(tensor=input_v.grad, dst=8)
                send_opt.wait()


            elif dist.get_rank() == 8:
                back_grad = torch.zeros([batch_size, 256, 8, 8], requires_grad=True)
                dist.recv(tensor=back_grad, src=7)
                input_v = queue.get()
                input_v.requires_grad_()
                output_v = layer(input_v)

                optimizer = optim.SGD(layer.parameters(), lr=0.01)
                optimizer.zero_grad()
                output_v.backward(back_grad)
                optimizer.step()
                send_opt = dist.isend(tensor=input_v.grad, dst=9)
                send_opt.wait()


            elif dist.get_rank() == 9:
                back_grad = torch.zeros([batch_size, 128, 16, 16], requires_grad=True)
                dist.recv(tensor=back_grad, src=8)
                input_v = queue.get()
                input_v.requires_grad_()
                output_v = layer(input_v)

                optimizer = optim.SGD(layer.parameters(), lr=0.01)
                optimizer.zero_grad()
                output_v.backward(back_grad)
                optimizer.step()
                send_opt = dist.isend(tensor=input_v.grad, dst=10)
                send_opt.wait()

            elif dist.get_rank() == 10:
                back_grad = torch.zeros([batch_size, 64, 32, 32], requires_grad=True)
                dist.recv(tensor=back_grad, src=9)
                input_v = queue.get()
                input_v.requires_grad_()
                output_v = layer(input_v)

                optimizer = optim.SGD(layer.parameters(), lr=0.01)
                optimizer.zero_grad()
                output_v.backward(back_grad)
                optimizer.step()
                send_opt = dist.isend(tensor=input_v.grad, dst=11)
                send_opt.wait()

            elif dist.get_rank() == 11:
                back_grad = torch.zeros([batch_size, 64, 32, 32], requires_grad=True)
                dist.recv(tensor=back_grad, src=10)
                print("rank" + str(dist.get_rank()) + " runing")
                if batch_count == 5:
                    stop_flag.value = 1
                    print("set stop.....")
                else:
                    print("batch_count:" + str(batch_count))
                    input_v = queue.get()
                    print("input_v")
                    input_v.requires_grad_()
                    output_v = layer(input_v)

                    optimizer = optim.SGD(layer.parameters(), lr=0.01)
                    optimizer.zero_grad()
                    output_v.backward(back_grad)
                    optimizer.step()
            batch_count += 1
        print("rank" + str(dist.get_rank()) + "is stop")
    except Exception as e:
        #traceback.print_exc()
        traceback.format_exc()
        return




def init_processes(fn, path, size, buffer_queues, layers, target_buffer, rank, stop_flag, trainloader):
    dist.init_process_group(backend='tcp', init_method=path, world_size=size, rank=rank)
    if rank == 0:
        fn(buffer_queues[0], layers[0], stop_flag, loader=trainloader,target_buffer=target_buffer)
    elif rank == 6:
        fn(buffer_queues[5], layers[5], stop_flag, target_buffer=target_buffer)
    else:
        fn(buffer_queues[rank if rank < 6 else (11 - rank)], layers[rank if rank < 6 else (11 - rank)], stop_flag)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-size', type=int, help='input the sum of node')
    parser.add_argument('-path', help='the path fo share file system')
    args = parser.parse_args()
    print("size:" + str(args.size))
    print("path:" + args.path)

    processes = []

    num_blocks = [2, 2, 2, 2]

    stop_flag = Value('i', 0)

    buffer_queues = []
    buffer_queues.append(Queue(100))
    buffer_queues.append(Queue(100))
    buffer_queues.append(Queue(100))
    buffer_queues.append(Queue(100))
    buffer_queues.append(Queue(100))
    buffer_queues.append(Queue(100))


    layers = []
    input_layer = ResInputLayer()
    input_layer.share_memory()

    layers.append(input_layer)

    block1 = ResBlockLayer(BasicBlock, 64, num_blocks[0], 1)
    block1.share_memory()
    layers.append(block1)
    in_plances = block1.get_in_plances()

    block2 = ResBlockLayer(BasicBlock, 128, num_blocks[1], 2, in_plances)
    block2.share_memory()
    layers.append(block2)
    in_plances = block2.get_in_plances()


    block3 = ResBlockLayer(BasicBlock, 256, num_blocks[2], 2, in_plances)
    block3.share_memory()
    layers.append(block3)
    in_plances = block3.get_in_plances()

    block4 = ResBlockLayer(BasicBlock, 512, num_blocks[3], 2, in_plances)
    block4.share_memory()
    layers.append(block4)
    in_plances = block4.get_in_plances()

    output_layer = ResOutputLayer(BasicBlock)
    output_layer.share_memory()
    layers.append(output_layer)

    target_buffer = Queue(100)


    #data processing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)


    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



    for rank in range(args.size):
        p = Process(target=init_processes, args=(run, args.path, args.size, buffer_queues, layers, target_buffer, rank, stop_flag, trainloader))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
