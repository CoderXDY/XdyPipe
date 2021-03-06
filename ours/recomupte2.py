import torch
import torch.distributed as dist
from torch import nn as nn
import argparse

from torch.multiprocessing import Queue, Event
from torch.multiprocessing import Process as P
from multiprocessing.managers import BaseManager as bm
from multiprocessing.dummy import Process
from multiprocessing.dummy import Queue as ThreadQueue
from multiprocessing.dummy import Semaphore
import torch.nn.functional as F
import torch.optim as optim
import logging
import time
from model.res import THResNet101Group20, THResNet101Group21, THResNet101Group1, THResNet50Group30, THResNet50Group31, THResNet50Group32, THResNet34Group0, THResNet34Group1, THResNet34Group2, THResNet18Group0, THResNet18Group1, THResNet18Group2
from model.vgg_module import VggLayer
from model.googlenet import GoogleNetGroup0, GoogleNetGroup1, GoogleNetGroup2
from model.dpn import  THDPNGroup0, THDPNGroup1, THDPNGroup2
import torchvision
import torchvision.transforms as transforms
from utils import progress_bar
import traceback
from queue import Empty, Full
import os
import psutil
import gc
import torch.backends.cudnn as cudnn
import numpy as np
import random




def get_left_right(tag):
    tag2rank = {
        -1: 0,
        0: 0,
        1: 1,
        2: 0,
        3: 0
    }
    left = tag2rank[tag - 1]
    right = tag2rank[tag + 1]
    return left, right


def transfer(tag, send_buf, shape):

    if shape == None:
        left, right = get_left_right(tag)
        send_opt = dist.isend(tensor=send_buf, dst=right)
        send_opt.wait()
        return None
    elif not torch.is_tensor(send_buf):
        left, right = get_left_right(tag)
        try:
            recv_buf = torch.zeros(shape)  # , dtype=torch.int8
            dist.recv(tensor=recv_buf, src=left)
        except RuntimeError as error:
            print("runtime error..")
            return None
        return recv_buf

    else:
        left, right = get_left_right(tag)
        send_opt = dist.isend(tensor=send_buf, dst=right)

        try:
            recv_buf = torch.zeros(shape)
            dist.recv(tensor=recv_buf, src=left)
        except RuntimeError as error:
            print("runtime error")
            return None
        send_opt.wait()
        return recv_buf






"""
1) add quan
2) whether to save quan value or not in local
3) init pipeline
4) acculate
"""

"""
get the node0 third batch_idx output
get the node1 second batch_idx output
get the node2 first batch_idx output
"""
def train(layer, logger, shapes, args, e, data_size, trainloader):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(layer.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    optimizer.zero_grad()
    layer.train()
    batch_idx = 0
    def backward_rank0(semaphore):
        batch_idx = 0
        grad_recv = torch.zeros(shapes[0])
        dist.recv(tensor=grad_recv, src=1)
        while True:
            print(" backwardbatch_idx:" + str(batch_idx))
            grad_recv = grad_recv.cuda(0)
            try:
                loss = outputs_queue.get(block=True, timeout=4)
                loss = loss.cuda(0)
            except Empty:
                print("empty........")
                break

            loss.backward(grad_recv)
            if batch_idx % args.ac == 0:
               # print("step: " + str(batch_idx))
                optimizer.step()
                optimizer.zero_grad()
            batch_idx += 1
            if data_size == batch_idx:
                print("eq...")
                break
            grad_recv = transfer(2, None, shapes[0])#shapes[0]
            print("backward send.....")
        print("backward end..")

    if dist.get_rank() == 0:
        outputs_queue = ThreadQueue(args.buffer_size)
        semaphore = Semaphore(args.buffer_size)
        back_process = Process(target=backward_rank0, args=(semaphore,))
        back_process.start()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            print("batch: " + str(batch_idx))
            inputs = inputs.cuda(0)
            outputs = layer(inputs)
            #outputs = outputs.cpu()
            outputs_queue.put(outputs)
            #ransfer(dist.get_rank(), outputs, None)
            transfer(dist.get_rank(), outputs.cpu(), None)
            print("send........")
        print("start to end....")

        back_process.join()
        e.set()
        print("end....")

    elif dist.get_rank() == 1:

        rec_val = None
        residual = None
        train_loss = 0
        correct = 0
        total = 0
        criterion.cuda(1)
        if not torch.is_tensor(rec_val):
            rec_val = torch.zeros(shapes[0])
            dist.recv(tensor=rec_val, src=0)
        for batch_idx, (_, targets) in enumerate(trainloader):
            rec_val = rec_val.cuda(1)
            rec_val.requires_grad_()
            outputs = layer(rec_val)
            # start to backward....
            targets = targets.cuda(1)
            loss = criterion(outputs, targets)
            loss.backward()

            if batch_idx % args.ac == 0:
                optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                progress_bar(batch_idx, data_size, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                optimizer.zero_grad()
            else:
                progress_bar(batch_idx, data_size, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            logger.error("train:" + str(train_loss / (batch_idx + 1)))
            acc_str = "tacc: %.3f" % (100. * correct / total,)
            logger.error(acc_str)
            if batch_idx == data_size - 1:
                transfer(dist.get_rank(), rec_val.grad.cpu(), None)
                continue
            rec_val = transfer(dist.get_rank(), rec_val.grad.cpu(), shapes[0])

        #print("\n start to end....")
        e.wait()
        print("end....")











def eval(layer, logger, e, save_event, data_size, testloader):
    criterion = nn.CrossEntropyLoss()
    criterion.cuda(2)
    layer.eval()
    with torch.no_grad():
        if dist.get_rank() == 0:
            for batch_idx, (inputs, targets) in enumerate(testloader):
                print('batch_idx: ' + str(batch_idx))
                inputs = inputs.cuda(0)
                outputs = layer(inputs)
                dist.send(tensor=outputs.cpu(), dst=1)
                print("send.....")

            e.wait()
        elif dist.get_rank() == 1:
            batch_idx = 0
            while data_size > batch_idx:
                print("batch_idx:" + str(batch_idx))
                rec_val = torch.zeros([100, 1024, 8, 8])  # difference model has difference shape
                dist.recv(tensor=rec_val, src=0)
                print("after recv....")
                outputs = layer(rec_val.cuda(1))
                dist.send(tensor=outputs.cpu(), dst=2)
                batch_idx += 1
                print("send...")

            e.wait()
        elif dist.get_rank() == 2:
            test_loss = 0
            correct = 0
            total = 0
            save_event.clear()
            global best_acc

            for batch_idx, (inputs, targets) in enumerate(testloader):
                rec_val = torch.zeros([100,  256, 8, 8])
                dist.recv(tensor=rec_val, src=1)
                outputs = layer(rec_val.cuda(2))
                targets = targets.cuda(2)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, data_size, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                logger.error("eval:" + str(test_loss / (batch_idx + 1)))
                acc_str = "eacc: %.3f" % (100. * correct / total,)
                logger.error(acc_str)
            time.sleep(1)
            acc = 100. * correct / total
            if acc > best_acc:
                best_acc = acc
                save_event.set()
            time.sleep(1)
            e.set()














def run(start_epoch, layer, shapes, args, targets_queue, global_event, epoch_event, save_event, train_size, test_size, trainloader, testloader):
    logger = logging.getLogger(args.model + '-f2-rank-' + str(dist.get_rank()))
    file_handler = logging.FileHandler(args.model + '-f2-rank-' + str(dist.get_rank()) + '.log')
    file_handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter(fmt='%(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    epoch_num = 200 - start_epoch
    global best_acc
    r = dist.get_rank()
    for epoch in range(start_epoch, start_epoch + epoch_num):
        print('Training epoch: %d' % epoch)
        set_seed(epoch + 1)
        train(layer, logger, shapes, args, epoch_event, train_size, trainloader)
        epoch_event.clear()
        # time.sleep(1)
        # print('Eval epoch: %d' % epoch)
        # eval(layer, logger, epoch_event, save_event, test_size, testloader)
        # epoch_event.clear()
        # if save_event.is_set():
        #     print('Saving..')
        #     state = {
        #         'net': layer.state_dict(),
        #         'acc': best_acc,
        #         'epoch': 0,
        #     }
        #     torch.save(state, './checkpoint/' + args.model + '-f2-rank-' + str(r) + '_ckpt.t7')
        # time.sleep(1)
    if r == 0 or r == 1:
        global_event.wait()
    elif r == 1:
        global_event.set()


def init_processes(run, layer, shapes, args, targets_queue, global_event, epoch_event, save_event, train_size,
        test_size, trainloader, testloader, rank):
    dist.init_process_group(backend='tcp', init_method=args.path, world_size=args.size, rank=rank)
    run(start_epoch, layer, shapes, args, targets_queue, global_event, epoch_event, save_event, train_size,
        test_size, trainloader, testloader)



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', help='the ip of master', default='89.72.2.41')
    parser.add_argument('-size', type=int, help='input the sum of node', default=3)
    parser.add_argument('-path', help='the path fo share file system',
                        default='file:///WORK/sysu_wgwu_4/xpipe/ours/temp')
    parser.add_argument('-rank', type=int, help='the rank of process')
    parser.add_argument('-batch_size', type=int, help='size of batch', default=64)
    parser.add_argument('-data_worker', type=int, help='the number of dataloader worker', default=0)
    parser.add_argument('-epoch', type=int, default=0)
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('-model', help='the path fo share file system')
    parser.add_argument('-buffer_size', type=int, help='size of batch', default=3)
    parser.add_argument('-port', type=int, default=5000)
    parser.add_argument('-ac', type=int, default=2)
    args = parser.parse_args()
    print("ip: " + args.ip)
    print("size: " + str(args.size))
    print("path: " + args.path)
    print("rank: " + str(args.rank))
    print("batch_size: " + str(args.batch_size))
    print("data_worker: " + str(args.data_worker))
    print("model: " + str(args.model))
    print("port: " + str(args.port))
    #torch.manual_seed(1)

    bm.register('get_epoch_event')
    bm.register('get_global_event')
    bm.register('get_grad_queue')
    bm.register('get_grad_queue2')
    bm.register('get_targets_queue')
    bm.register('get_save_event')
    bm.register('get_backward_event')
    bm.register('get_start_thread_event')
    bm.register('get_start_thread_event2')
    m = bm(address=(args.ip, args.port), authkey=b'xpipe')
    m.connect()
    global_event = m.get_global_event()
    epoch_event = m.get_epoch_event()
    grad_queue = m.get_grad_queue()
    targets_queue = m.get_targets_queue()
    save_event = m.get_save_event()
    start_event = m.get_start_thread_event()
    grad_queue2 = m.get_grad_queue2()
    start_event2 = m.get_start_thread_event2()

    """
        difference model
        """
    #node_cfg_0 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512]
    #node_cfg_1 = [512, 512, 'M', 512, 512,512, 512, 'M']

    #vgg19
    #shapes = [[args.batch_size, 512, 4, 4]]
    #res101
    shapes =[[args.batch_size, 1024, 8, 8]]
    if args.rank == 0:
        #layer = VggLayer(node_cfg_0)
        layer = THResNet101Group20()
        layer.cuda(0)
        layer.share_memory()
    elif args.rank == 1:
        #layer = VggLayer(node_cfg_1, node_cfg_0[-1] if node_cfg_0[-1] != 'M' else node_cfg_0[-2], last_flag=True)
        #layer = GoogleNetGroup1()
        #layer = VggLayer(node_cfg_1, node_cfg_0[-1] if node_cfg_0[-1] != 'M' else node_cfg_0[-2])
        #layer = THDPNGroup1()
        #layer = THResNet50Group31()
        #layer = THResNet34Group1()
        #layer = THResNet18Group1()
        layer = THResNet101Group21()
        layer.cuda(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #layer.share_memory()
    cudnn.benchmark = True

    best_acc = 0.0
    start_epoch = 0

    if True:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform_train)

        # pin memory
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.data_worker, drop_last=True)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                                 num_workers=args.data_worker, drop_last=True)
        train_size = len(trainloader)
        test_size = len(testloader)


    if args.epoch != 0:
        start_epoch = args.epoch



    if args.rank == 1:
        dist.init_process_group(backend='tcp', init_method=args.path, world_size=args.size, rank=args.rank)
        run(start_epoch, layer, shapes, args, targets_queue, global_event, epoch_event, save_event, train_size,
            test_size, trainloader, testloader)
    elif args.rank == 0:
        p0 = P(target=init_processes, args=(run, start_epoch, layer, shapes, args, targets_queue, global_event, epoch_event, save_event, train_size, test_size, trainloader, testloader, 0))
        p1 = P(target=init_processes, args=(run, start_epoch, layer, shapes, args, targets_queue, global_event, epoch_event, save_event, train_size, test_size, trainloader, testloader, 2))