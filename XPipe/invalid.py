import torch
from torch.autograd import Variable as V
import torch.distributed as dist
from torch import nn as nn
import argparse
from torch.multiprocessing import Process, Queue, Value, Event
from multiprocessing.managers import BaseManager as bm
import torch.nn.functional as F
import torch.optim as optim
import logging
import time
from res import BasicBlock, Bottleneck, ResInputLayer, ResBlockLayer, ResOutputLayer
import torchvision
import torchvision.transforms as transforms
from utils import progress_bar
import traceback
from queue import Empty, Full
import os
import psutil

"""
 pipeline ResNet script for Tianhe-2  memory optim

"""

def train(queue, layer, e, args, loader=None, target_buffer=None):

    logger = logging.getLogger('rank-' +str(dist.get_rank()))
    file_handler = logging.FileHandler('/WORK/sysu_wgwu_4/xpipe/XPipe/rank-' + str(dist.get_rank()) + '.log')
    file_handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter(fmt='%(levelname)s:%(asctime)s | pricess_id-%(process)d | %(funcName)s->%(lineno)d | %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    batch_size = args.batch_size
    package_size = args.package_size
    send_num = args.send_num
    epoch = 0
    max_epoch = args.epoch

    all_loss = 0
    batch_idx = 0
    total = 0
    correct = 0

    access_stop_flag = False

    queue_wait = 5

    point = 0
    time_sleep = 6



    if loader is not None and (dist.get_rank() == 0 or dist.get_rank() == 6):
        data_iter = iter(loader)




    package = None
    rec_val = None

    if dist.get_rank() == 0 or dist.get_rank() == 1 or dist.get_rank() == 9 or dist.get_rank() == 10:
        package = torch.zeros([package_size, batch_size, 64, 32, 32], requires_grad=True)
    elif dist.get_rank() == 2 or dist.get_rank() == 8:
        package = torch.zeros([package_size, batch_size, 128, 16, 16], requires_grad=True)
    elif dist.get_rank() == 3 or dist.get_rank() == 7:
        package = torch.zeros([package_size, batch_size, 256, 8, 8], requires_grad=True)
    elif dist.get_rank() == 4 or dist.get_rank() == 6:
        package = torch.zeros([package_size, batch_size, 512, 4, 4], requires_grad=True)
    if package is not None:
        package.share_memory_()

    if dist.get_rank() == 0:
        input_v_pack = torch.zeros([package_size, batch_size, 3, 32, 32], requires_grad=True)
        input_v_pack.share_memory_()
    elif dist.get_rank() == 1 or dist.get_rank() == 2:
        rec_val = torch.zeros([package_size, batch_size, 64, 32, 32], requires_grad=True)
    elif dist.get_rank() == 3:
        rec_val = torch.zeros([package_size, batch_size, 128, 16, 16], requires_grad=True)
    elif dist.get_rank() == 4:
        rec_val = torch.zeros([package_size, batch_size, 256, 8, 8], requires_grad=True)
    elif dist.get_rank() == 5:
        rec_val = torch.zeros([package_size, batch_size, 512, 4, 4], requires_grad=True)
    if rec_val is not None:
        rec_val.share_memory_()

    try:
        while True:
            if dist.get_rank() == 0:
                try:
                    for count in range(package_size):
                        input_v, target_v = next(data_iter)
                        input_v_pack[count] = input_v
                        output_v = layer(input_v)
                        package[count] = output_v
                    try:
                        queue.put(input_v_pack, timeout=queue_wait)
                    except Full as full:
                        logger.error("queue full.......")
                        time.sleep(time_sleep)
                        queue.put(input_v_pack)
                        dist.send(tensor=package, dst=1)
                        logger.error('full wait and rank 0 send.....')
                        continue
                except StopIteration as stop_e:
                    if epoch < max_epoch:
                        logger.error('rank-%s: epoch-%s start...', str(dist.get_rank()), str(epoch))
                        epoch += 1
                        data_iter = iter(loader)
                        continue
                    else:
                        logger.error('iteration end successfully......')
                        #print("iter end......")
                        dist.send(tensor=torch.zeros(1), dst=1)
                        e.wait()
                        break
                dist.send(tensor=package, dst=1)
                logger.error('rank 0 send.....')


            elif dist.get_rank() == 1:
                try:
                    dist.recv(tensor=rec_val, src=0)
                except RuntimeError as error:
                    dist.send(tensor=torch.zeros(1), dst=2)
                    e.wait()
                    break


                for count in range(package_size):
                    one_batch = rec_val[count]
                    output_v = layer(one_batch)
                    package[count] = output_v
                try:
                    queue.put(rec_val, timeout=queue_wait)
                except Full as full:
                    time.sleep(time_sleep)
                    queue.put(rec_val)
                    dist.send(tensor=package, dst=2)
                    logger.error('full wait and rank 1 send....')
                    continue
                dist.send(tensor=package, dst=2)
                logger.error('rank 1 send....')

            elif dist.get_rank() == 2:
                try:

                    dist.recv(tensor=rec_val, src=1)
                except RuntimeError as error:
                    dist.send(tensor=torch.zeros(1), dst=3)
                    e.wait()
                    break

                for count in range(package_size):
                    one_batch = rec_val[count]
                    output_v = layer(one_batch)
                    package[count] = output_v
                try:
                    queue.put(rec_val, timeout=queue_wait)
                except Full as full:
                    time.sleep(time_sleep)
                    queue.put(rec_val)
                    dist.send(tensor=package, dst=3)
                    logging.error('full wait and rank 2 send.....')
                    continue
                dist.send(tensor=package, dst=3)
                logging.error('rank 2 send.....')
            elif dist.get_rank() == 3:
                try:

                    dist.recv(tensor=rec_val, src=2)
                except RuntimeError as error:
                    dist.send(tensor=torch.zeros(1), dst=4)
                    e.wait()
                    break


                for count in range(package_size):
                    one_batch = rec_val[count]
                    output_v = layer(one_batch)
                    package[count] = output_v
                try:
                    queue.put(rec_val, timeout=queue_wait)
                except Full as full:
                    time.sleep(time_sleep)
                    queue.put(rec_val)
                    dist.send(tensor=package, dst=4)
                    logging.error('full wait rank 3 send.......')
                    continue
                dist.send(tensor=package, dst=4)
                logging.error('rank 3 send.......')

            elif dist.get_rank() == 4:
                try:

                    dist.recv(tensor=rec_val, src=3)
                except RuntimeError as error:
                    dist.send(tensor=torch.zeros(1), dst=5)
                    e.wait()
                    break


                for count in range(package_size):
                    one_batch = rec_val[count]
                    output_v = layer(one_batch)
                    package[count] = output_v
                try:
                    queue.put(rec_val, timeout=queue_wait)
                except Full as full:
                    time.sleep(time_sleep)
                    queue.put(rec_val)
                    dist.send(tensor=package, dst=5)
                    logger.error('full wait and rank 4 send.....')
                    continue
                dist.send(tensor=package, dst=5)
                logger.error('rank 4 send.....')

            elif dist.get_rank() == 5:
                try:

                    dist.recv(tensor=rec_val, src=4)
                except RuntimeError as error:
                    dist.send(tensor=torch.zeros(1), dst=6)
                    e.wait()
                    break

                try:
                    queue.put(rec_val, timeout=queue_wait)
                except:
                    time.sleep(time_sleep)
                    queue.put(rec_val)
                    dist.send(tensor=torch.randn(2), dst=6)
                    logger.error('full wait and rank 5 send......')
                    continue

                dist.send(tensor=torch.randn(2), dst=6)

                logger.error('rank 5 send......')

            elif dist.get_rank() == 6:
                logger.error("rank 6 run....")
                try:
                    if not access_stop_flag:
                        rec_val = torch.zeros(2)
                        dist.recv(tensor=rec_val, src=5)
                except RuntimeError as error:
                    access_stop_flag = True
                finally:
                    try:
                        rec_pack = queue.get(block=True, timeout=queue_wait)
                    except Empty as empty:
                        logger.error('rank 6 prepare to end....')
                        dist.send(tensor=torch.zeros(1), dst=7)
                        e.wait()
                        break

                    target_v_pack = torch.zeros([package_size, batch_size], dtype=torch.long)
                    for count in range(package_size):
                        _, target_temp = next(data_iter)
                        target_v_pack[count] = target_temp

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
                        _, predicted = output_v.max(1)
                        total += target_v.size(0)
                        correct += predicted.eq(target_v).sum().item()
                        logger.error('batch_idx: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (batch_idx, all_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                        #progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (all_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                        package[count] = input_v.grad
                    dist.send(tensor=package, dst=7)
                    #dist.isend(tensor=package, dst=7)
                    logger.error('rank 6 send.....')
                    #print("rank 6 send....")
            elif dist.get_rank() == 7:
                try:
                    if not access_stop_flag:
                        back_grad_pack = torch.zeros([package_size, batch_size, 512, 4, 4], requires_grad=True)
                        dist.recv(tensor=back_grad_pack, src=6)
                except RuntimeError as error:
                    access_stop_flag = True
                finally:
                    try:
                        input_v_pack = queue.get(block=True, timeout=queue_wait)
                    except Empty as empty:
                        logger.error('rank 7 prepare to end.....')
                        #print("rank 7 prepare to end.....")
                        dist.send(tensor=torch.zeros(1), dst=8)
                        e.wait()
                        break

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
                    dist.send(tensor=package, dst=8)
                    #dist.isend(tensor=package, dst=8)
                    logger.error('rank 7 send....')
                    #print("rank 7 send.....")
            elif dist.get_rank() == 8:
                try:
                    if not access_stop_flag:
                        back_grad_pack = torch.zeros([package_size, batch_size, 256, 8, 8], requires_grad=True)
                        dist.recv(tensor=back_grad_pack, src=7)
                except RuntimeError as error:
                    access_stop_flag = True
                finally:
                    try:
                        input_v_pack = queue.get(block=True, timeout=queue_wait)
                    except Empty as empty:
                        logger.error('rank 8 prepare to end.....')
                        #print("rank 8 prepare to end.....")
                        dist.send(tensor=torch.zeros(1), dst=9)
                        e.wait()
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
                        package[count] = input_v.grad
                    #dist.isend(tensor=package, dst=9)
                    dist.send(tensor=package, dst=9)
                    logger.error('rank 8 send.....')
                    #print("rank 8 send......")

            elif dist.get_rank() == 9:
                try:
                    if not access_stop_flag:
                        back_grad_pack = torch.zeros([package_size, batch_size, 128, 16, 16], requires_grad=True)
                        dist.recv(tensor=back_grad_pack, src=8)
                except RuntimeError as error:
                    access_stop_flag = True
                finally:
                    try:
                        input_v_pack = queue.get(block=True, timeout=queue_wait)
                    except Empty as empty:
                        logger.error('rank 9 prepare to end....')
                        #print("rank 9 prepare to end......")
                        dist.send(tensor=torch.zeros(1), dst=10)
                        e.wait()
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
                        package[count] = input_v.grad
                    dist.send(tensor=package, dst=10)
                    #dist.isend(tensor=package, dst=10)
                    logger.error('rank 9 send....')
                    #print("rank 9 send......")
            elif dist.get_rank() == 10:
                try:
                    if not access_stop_flag:
                        back_grad_pack = torch.zeros([package_size, batch_size, 64, 32, 32], requires_grad=True)
                        dist.recv(tensor=back_grad_pack, src=9)
                except RuntimeError as error:
                    access_stop_flag = True
                finally:
                    try:
                        input_v_pack = queue.get(block=True, timeout=queue_wait)
                    except Empty as empty:
                        logger.error('rank 10 prepare to end.....')
                        #print("rank 10 prepare to end.....")
                        dist.send(tensor=torch.zeros(1), dst=11)
                        e.wait()
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
                        package[count] = input_v.grad
                    dist.send(tensor=package, dst=11)
                    #dist.isend(tensor=package, dst=11)
                    logger.error('rank 10 send......')
                    #print("rank 10 send.....")
            elif dist.get_rank() == 11:
                try:
                    if not access_stop_flag:
                        back_grad_pack = torch.zeros([package_size, batch_size, 64, 32, 32], requires_grad=True)
                        dist.recv(tensor=back_grad_pack, src=10)
                except RuntimeError as error:
                    access_stop_flag = True
                finally:
                    try:
                        input_v_pack = queue.get(block=True, timeout=queue_wait)
                    except Empty as empty:
                        logger.info('rank 10 start to stop....')
                        #print("rank 10 start to stop")
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
            if point % 10 == 0:
                mem = psutil.virtual_memory()
                swp = psutil.swap_memory()
                cpu = psutil.cpu_times()
                netio = psutil.net_io_counters()
                pid = os.getpid()
                p = psutil.Process(pid)
                logger.error("record-" + str(point) + "....")
                logger.error(str(cpu))
                logger.error(str(mem))
                logger.error(str(swp))
                logger.error(str(netio))
                logger.error("process status:" + str(p.status()))
                logger.error(str(p.cpu_times()))
                logger.error(str(p.memory_info()))
            point += 1
        logger.info('rank-%s stop....', str(dist.get_rank()))
        #print("rank-" + str(dist.get_rank()) + " stop.......")
    except Exception as e:
        logger.error('rank-' + str(dist.get_rank()) + ' fail: ', exc_info=True)
        #print(e)
        return





def run(queue, layer, e, args, train_loader=None, test_loader=None, target_buffer=None):
    layer.train()
    train(queue, layer, e, args, train_loader, target_buffer)
    # for epoch in range(4):
    # print(str(dist.get_rank()) + "-train iter-" + str(epoch))
    # layer.train()
    # train(queue, layer, e, train_loader, target_buffer)
    # print("test iter-" + str(epoch))
    # layer.eval()
    # with torch.no_grad():
    # test(queue, layer, e, test_loader, target_buffer)


def init_processes(fn, args, queue, layer, rank, e):
    print("init process-" + str(rank) + "....")
    dist.init_process_group(backend='tcp', init_method=args.path, world_size=args.size, rank=rank)

    if rank == 0 or rank == 6:

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.data_worker, drop_last=True)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.data_worker, drop_last=True)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    target_buffer = None
    if rank == 0:
        fn(queue, layer, e, args, train_loader=trainloader, test_loader=testloader,
           target_buffer=target_buffer)
    elif rank == 6:
        fn(queue, layer, e, args, train_loader=trainloader, target_buffer=target_buffer)
    else:
        fn(queue, layer, e, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', help='the ip of master',default='89.72.2.41')
    parser.add_argument('-size', type=int, help='input the sum of node', default=12)
    parser.add_argument('-path', help='the path fo share file system')
    parser.add_argument('-rank', type=int, help='the rank of process')

    parser.add_argument('-buffer_size', type=int, help='the size of buffer queue caching the batch data', default=16)

    parser.add_argument('-layer_type', type=int, help='type of layer: input:0, block:1, output:2')

    parser.add_argument('-basic', help='if True, using basicblock for ResNet, else use Bottleneck')
    parser.add_argument('-out_plane', type=int, help='out_plane for cnn')
    parser.add_argument('-num_block', type=int, help='number of block (BasicBlock or Bottleneck)')
    parser.add_argument('-stride',  type=int, help='stride.....')
    parser.add_argument('-in_plane', type=int, help='in_plane for cnn')


    parser.add_argument('-batch_size', type=int, help='size of batch')
    parser.add_argument('-data_worker', type=int, help='the number of dataloader worker')
    parser.add_argument('-epoch', type=int)
    parser.add_argument('-package_size', type=int)
    parser.add_argument('-send_num', type=int)



    args = parser.parse_args()
    print("ip: " + args.ip)
    print("size: " + str(args.size))
    print("path: " + args.path)
    print("rank: " + str(args.rank))
    print("buffer_size: " + str(args.buffer_size))
    print("layer_type: " + str(args.layer_type))
    print("basic: " + args.basic)
    print("out_plane: " + str(args.out_plane))
    print("num_block: " + str(args.num_block))
    print("stride: " + str(args.stride))
    print("in_plane: " + str(args.in_plane))
    print("batch_size: " + str(args.batch_size))
    print("data_worker: " + str(args.data_worker))

    time.sleep(2)

    bm.register('get_event')
    #bm.register('get_queue')
    m = bm(address=(args.ip, 5000), authkey=b'xpipe')
    m.connect()
    e = m.get_event()

    # target_queue = None
    #
    # if args.rank == 0 or args.rank == 6:
    #     target_queue = m.get_queue()
    #     target_queue.put(0)
    #     a = target_queue.get()
    #     print("a:" + a)
    queue = Queue(args.buffer_size)

    if args.layer_type == 0:
        layer = ResInputLayer()
    elif args.layer_type == 1:
        layer = ResBlockLayer(BasicBlock if args.basic == 'True' else Bottleneck,
                              args.out_plane,
                              args.num_block,
                              args.stride,
                              args.in_plane)
    elif args.layer_type == 2:
        layer = ResOutputLayer(BasicBlock if args.basic == 'True' else Bottleneck)
    layer.share_memory()

    f_p = Process(target=init_processes, args=(run, args, queue, layer, args.rank, e))
    f_p.start()
    b_p = Process(target=init_processes, args=(run, args, queue, layer, (11 - args.rank), e))
    b_p.start()
    f_p.join()
    b_p.join()