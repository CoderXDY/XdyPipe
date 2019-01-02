import torch
from torch.autograd import Variable as V
import torch.distributed as dist
from torch import nn as nn
import argparse
from torch.multiprocessing import Process, Queue, Value, Event
import torch.nn.functional as F
import torch.optim as optim
import logging
import time
from res import BasicBlock, Bottleneck, ResInputLayer, ResBlockLayer, ResOutputLayer
import torchvision
import torchvision.transforms as transforms
from utils import progress_bar
import traceback
from queue import Empty
import os

def train(queue, layer, e, loader=None, target_buffer=None):

    logger = logging.getLogger('rank-' +str(dist.get_rank()))
    file_handler = logging.FileHandler('rank-' + str(dist.get_rank()) + '.log')
    file_handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter(fmt='%(levelname)s:%(asctime)s | pricess_id-%(process)d | %(funcName)s->%(lineno)d | %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    batch_size = 128

    all_loss = 0
    batch_idx = 0
    total = 0
    correct = 0

    epoch = 0

    access_stop_flag = False


    package_size = 4

    send_num = 4


    logger.debug('rank - ' + str(dist.get_rank()) + ' start........')

    if loader is not None and dist.get_rank() == 0:
        data_iter = iter(loader)
    try:
        while True:

            if dist.get_rank() == 0:
                package = torch.zeros([package_size, batch_size, 64, 32, 32], requires_grad=True)
                try:
                    input_v_pack = torch.zeros([package_size, batch_size, 3, 32, 32], requires_grad=True)
                    target_v_pack = torch.zeros([package_size, batch_size], dtype=torch.long)
                    for count in range(package_size):
                        input_v, target_v = next(data_iter)
                        input_v_pack[count] = input_v
                        target_v_pack[count] = target_v
                        output_v = layer(input_v)
                        package[count] = output_v
                        logger.error('rank 0 iter package....')
                    input_v_pack.share_memory_()
                    queue.put(input_v_pack)
                    target_v_pack.share_memory_()
                    target_buffer.put(target_v_pack)
                except StopIteration as stop_e:
                    if epoch < 2:
                        logger.info('rank-%s: epoch-%s start...', str(dist.get_rank()), str(epoch))
                        epoch += 1
                        data_iter = iter(loader)
                        continue
                    else:
                        logger.info('iteration end successfully......')
                        dist.send(tensor=torch.zeros(1), dst=1)
                        e.wait()
                        break
                except Exception as e:
                    logger.error('rank-' + str(dist.get_rank()) + ' faild' , exc_info=True)
                    break
                send_opt = dist.isend(tensor=package, dst=1)
                if queue.qsize() > send_num:
                    send_opt.wait()
                logger.error('rank 0 send.....')

            elif dist.get_rank() == 1:
                try:
                    rec_val = torch.zeros([package_size, batch_size, 64, 32, 32], requires_grad=True)
                    dist.recv(tensor=rec_val, src=0)
                except RuntimeError as error:
                    logger.error('rank 1 failed', exc_info=True)
                    logger.error('rank 1 wait..............')
                    dist.send(tensor=torch.zeros(1), dst=2)
                    e.wait()
                    break
                rec_val.share_memory_()
                queue.put(rec_val)

                package = torch.zeros([package_size, batch_size, 64, 32, 32], requires_grad=True)
                for count in range(package_size):
                    one_batch = rec_val[count]
                    output_v = layer(one_batch)
                    package[count] = output_v
                send_opt = dist.isend(tensor=package, dst=2)
                if queue.qsize() > send_num:
                    send_opt.wait()
                logger.error('rank 1 send....')

            elif dist.get_rank() == 2:
                try:
                    rec_val = torch.zeros([package_size, batch_size, 64, 32, 32], requires_grad=True)
                    dist.recv(tensor=rec_val, src=1)
                except RuntimeError as error:
                    logger.error('rank 2 failed', exc_info=True)
                    logger.error('rank 2 wait..............')
                    dist.send(tensor=torch.zeros(1), dst=3)
                    e.wait()
                    break
                except Exception as ee:
                    logger.error('rank 2 failed', exc_info=True)
                    break
                rec_val.share_memory_()
                queue.put(rec_val)

                package = torch.zeros([package_size, batch_size, 128, 16, 16], requires_grad=True)
                for count in range(package_size):
                    one_batch = rec_val[count]
                    output_v = layer(one_batch)
                    package[count] = output_v
                send_opt = dist.isend(tensor=package, dst=3)
                if queue.qsize() > send_num:
                    send_opt.wait()
                logging.debug('rank 2 send.....')
                print('rank 2 send.........')
            elif dist.get_rank() == 3:
                try:
                    rec_val = torch.zeros([package_size, batch_size, 128, 16, 16], requires_grad=True)
                    dist.recv(tensor=rec_val, src=2)
                except RuntimeError as error:
                    logger.error('rank 3 failed', exc_info=True)
                    logger.error('rank 3 wait..............')
                    dist.send(tensor=torch.zeros(1), dst=4)
                    e.wait()
                    break
                rec_val.share_memory_()
                queue.put(rec_val)

                package = torch.zeros([package_size, batch_size, 256, 8, 8], requires_grad=True)
                for count in range(package_size):
                    one_batch = rec_val[count]
                    output_v = layer(one_batch)
                    package[count] = output_v
                send_opt = dist.isend(tensor=package, dst=4)
                if queue.qsize() > send_num:
                    send_opt.wait()
                logging.debug('rank 3 send.......')
                print('rank 3 send.....')
            elif dist.get_rank() == 4:
                try:
                    rec_val = torch.zeros([package_size, batch_size, 256, 8, 8], requires_grad=True)
                    dist.recv(tensor=rec_val, src=3)
                except RuntimeError as error:
                    logger.error('rank 4 failed', exc_info=True)
                    logger.error('rank 4 wait..............')
                    dist.send(tensor=torch.zeros(1), dst=5)
                    e.wait()
                    break
                rec_val.share_memory_()
                queue.put(rec_val)

                package = torch.zeros([package_size, batch_size, 512, 4, 4], requires_grad=True)
                for count in range(package_size):
                    one_batch = rec_val[count]
                    output_v = layer(one_batch)
                    package[count] = output_v
                send_opt = dist.isend(tensor=package, dst=5)
                if queue.qsize() > send_num:
                    send_opt.wait()
                logger.error('rank 4 send.....')
            elif dist.get_rank() == 5:
                try:
                    rec_val = torch.zeros([package_size, batch_size, 512, 4, 4], requires_grad=True)
                    dist.recv(tensor=rec_val, src=4)
                except RuntimeError as error:
                    logger.error('rank 5 failed', exc_info=True)
                    logger.error('rank 5 wait..............')
                    dist.send(tensor=torch.zeros(1), dst=6)
                    e.wait()
                    break
                rec_val.share_memory_()
                queue.put(rec_val)

                send_opt = dist.isend(tensor=torch.randn(2), dst=6)
                if queue.qsize() > send_num:
                    send_opt.wait()
                logger.error('rank 5 send......')
            elif dist.get_rank() == 6:
                try:
                    if not access_stop_flag:
                        rec_val = torch.zeros(2)
                        dist.recv(tensor=rec_val, src=5)
                except RuntimeError as error:
                    access_stop_flag = True
                finally:
                    try:
                        rec_pack = queue.get(block=True, timeout=2)
                    except Empty as empty:
                        logger.error('rank 6 wait......')
                        dist.send(tensor=torch.zeros(1), dst=7)
                        e.wait()
                        break
                    package = torch.zeros([package_size, batch_size, 512, 4, 4], requires_grad=True)
                    target_v_pack = target_buffer.get()
                    for count in range(package_size):
                        input_v = rec_pack[count]
                        input_v.requires_grad_()
                        output_v = layer(input_v)

                        optimizer = optim.SGD(layer.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

                        optimizer.zero_grad()

                        criterion = nn.CrossEntropyLoss()
                        target_v = target_v_pack[count]
                        batch_idx += 1
                        loss = criterion(output_v, target_v)
                        loss.backward()

                        optimizer.step()

                        all_loss += loss.item()
                        logger.error(str(dist.get_rank()) + "-train-loss:" + str(loss.item()))
                        _, predicted = output_v.max(1)
                        total += target_v.size(0)
                        correct += predicted.eq(target_v).sum().item()

                        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (all_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                        package[count] = input_v.grad

                    send_opt = dist.isend(tensor=package, dst=7)
                    logger.error('rank 6 send.....')
            elif dist.get_rank() == 7:
                try:
                    if not access_stop_flag:
                        back_grad_pack = torch.zeros([package_size, batch_size, 512, 4, 4], requires_grad=True)
                        dist.recv(tensor=back_grad_pack, src=6)
                except RuntimeError as error:
                    access_stop_flag = True
                finally:
                    try:
                        input_v_pack = queue.get(block=True, timeout=2)
                    except Empty as empty:
                        logger.error('rank 7 wait.....')
                        dist.send(tensor=torch.zeros(1), dst=8)
                        e.wait()
                        break
                    package = torch.zeros([package_size, batch_size, 256, 8, 8], requires_grad=True)
                    for count in range(package_size):
                        input_v = input_v_pack[count]
                        input_v.requires_grad_()
                        back_grad = back_grad_pack[count]# requires_grad == True????????
                        output_v = layer(input_v)

                        optimizer = optim.SGD(layer.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
                        optimizer.zero_grad()
                        output_v.backward(back_grad)
                        optimizer.step()
                        package[count] = input_v.grad
                    send_opt = dist.isend(tensor=package, dst=8)
                    logger.error('rank 7 send....')
            elif dist.get_rank() == 8:
                try:
                    if not access_stop_flag:
                        back_grad_pack = torch.zeros([package_size, batch_size, 256, 8, 8], requires_grad=True)
                        dist.recv(tensor=back_grad_pack, src=7)
                except RuntimeError as error:
                    access_stop_flag = True
                finally:
                    try:
                        input_v_pack = queue.get(block=True, timeout=2)
                    except Empty as empty:
                        logger.error('rank 8 wait.....')
                        dist.send(tensor=torch.zeros(1), dst=9)
                        e.wait()
                        break
                    package = torch.zeros([package_size, batch_size, 128, 16, 16], requires_grad=True)
                    for count in range(package_size):
                        input_v = input_v_pack[count]
                        input_v.requires_grad_()
                        back_grad = back_grad_pack[count]
                        output_v = layer(input_v)

                        optimizer = optim.SGD(layer.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
                        optimizer.zero_grad()
                        output_v.backward(back_grad)
                        optimizer.step()
                        package[count] = input_v.grad
                    send_opt = dist.isend(tensor=package, dst=9)
                    logger.error('rank 8 send.....')

            elif dist.get_rank() == 9:
                try:
                    if not access_stop_flag:
                        back_grad_pack = torch.zeros([package_size, batch_size, 128, 16, 16], requires_grad=True)
                        dist.recv(tensor=back_grad_pack, src=8)
                except RuntimeError as error:
                    access_stop_flag = True
                finally:
                    try:
                        input_v_pack = queue.get(block=True, timeout=2)
                    except Empty as empty:
                        logger.error('rank 9 wait....')
                        dist.send(tensor=torch.zeros(1), dst=10)
                        e.wait()
                        break
                    package = torch.zeros([package_size, batch_size, 64, 32, 32], requires_grad=True)
                    for count in range(package_size):
                        input_v = input_v_pack[count]
                        input_v.requires_grad_()
                        back_grad = back_grad_pack[count]
                        output_v = layer(input_v)
                        optimizer = optim.SGD(layer.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
                        optimizer.zero_grad()
                        output_v.backward(back_grad)
                        optimizer.step()
                        package[count] = input_v.grad
                    send_opt = dist.isend(tensor=package, dst=10)
                    logger.error('rank 9 send....')
            elif dist.get_rank() == 10:
                try:
                    if not access_stop_flag:
                        back_grad_pack = torch.zeros([package_size, batch_size, 64, 32, 32], requires_grad=True)
                        dist.recv(tensor=back_grad_pack, src=9)
                except RuntimeError as error:
                    access_stop_flag = True
                finally:
                    try:
                        input_v_pack = queue.get(block=True, timeout=3)
                    except Empty as empty:
                        logger.error('rank 10 wait.....')
                        dist.send(tensor=torch.zeros(1), dst=11)
                        e.wait()
                        break
                    package = torch.zeros([package_size, batch_size, 64, 32, 32], requires_grad=True)
                    for count in range(package_size):
                        input_v = input_v_pack[count]
                        input_v.requires_grad_()
                        back_grad = back_grad_pack[count]
                        output_v = layer(input_v)
                        optimizer = optim.SGD(layer.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
                        optimizer.zero_grad()
                        output_v.backward(back_grad)
                        optimizer.step()
                        package[count] = input_v.grad
                    send_opt=dist.isend(tensor=package, dst=11)
                    logger.error('rank 10 send......')

            elif dist.get_rank() == 11:
                try:
                    if not access_stop_flag:
                        back_grad_pack = torch.zeros([package_size, batch_size, 64, 32, 32], requires_grad=True)
                        dist.recv(tensor=back_grad_pack, src=10)
                        print('rank 10 recv....')
                except RuntimeError as error:
                    access_stop_flag = True
                    logger.error('rank 11 rumtime error', exe_info=True)
                finally:
                    try:
                        input_v_pack = queue.get(block=True, timeout=2)
                        print("rank 10 queue get.......")
                    except Empty as empty:
                        logger.info('rank 10 start to stop....')
                        e.set()
                        break
                    for count in range(package_size):
                        input_v = input_v_pack[count]
                        input_v.requires_grad_()
                        back_grad = back_grad_pack[count]
                        output_v = layer(input_v)

                        optimizer = optim.SGD(layer.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
                        optimizer.zero_grad()
                        output_v.backward(back_grad)
                        optimizer.step()
        logger.info('rank-%s stop....', str(dist.get_rank()))
    except Exception as e:
        logger.error('rank-' + str(dist.get_rank()) + 'faild', exc_info=True)
        return





def run(queue, layer, e, train_loader=None, test_loader=None, target_buffer=None):
    layer.train()
    train(queue, layer, e, train_loader, target_buffer)
    # for epoch in range(4):
    # print(str(dist.get_rank()) + "-train iter-" + str(epoch))
    # layer.train()
    # train(queue, layer, e, train_loader, target_buffer)
    # print("test iter-" + str(epoch))
    # layer.eval()
    # with torch.no_grad():
    # test(queue, layer, e, test_loader, target_buffer)


def init_processes(fn, path, size, buffer_queues, layers, target_buffer, rank, e, trainloader, testloader):
    print("init process-" + str(rank) + "....")
    dist.init_process_group(backend='tcp', init_method=path, world_size=size, rank=rank)
    if rank == 0:
        fn(buffer_queues[0], layers[0], e, train_loader=trainloader, test_loader=testloader,
           target_buffer=target_buffer)
    elif rank == 6:
        fn(buffer_queues[5], layers[5], e, train_loader=trainloader, target_buffer=target_buffer)
    else:
        fn(buffer_queues[rank if rank < 6 else (11 - rank)], layers[rank if rank < 6 else (11 - rank)], e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-size', type=int, help='input the sum of node')
    parser.add_argument('-path', help='the path fo share file system')
    args = parser.parse_args()
    print("size:" + str(args.size))
    print("path:" + args.path)

    processes = []

    num_blocks = [2, 2, 2, 2]

    # stop_flag = Value('i', 0)
    e = Event()

    buffer_queues = []
    buffer_queues.append(Queue(400))
    buffer_queues.append(Queue(400))
    buffer_queues.append(Queue(400))
    buffer_queues.append(Queue(400))
    buffer_queues.append(Queue(400))
    buffer_queues.append(Queue(400))

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

    # data processing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1, drop_last=True)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1, drop_last=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for rank in range(args.size):
        p = Process(target=init_processes, args=(
        run, args.path, args.size, buffer_queues, layers, target_buffer, rank, e, trainloader, testloader))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
