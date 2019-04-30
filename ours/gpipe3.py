import torch
import torch.distributed as dist
from torch import nn as nn
import argparse
import queue as Q
from torch.multiprocessing import Queue, Event
from multiprocessing.managers import BaseManager as bm
from multiprocessing.dummy import Process
from multiprocessing.dummy import Queue as ThreadQueue
from multiprocessing.dummy import Semaphore
import torch.nn.functional as F
import torch.optim as optim
import logging
import time
from model.res import THResNet101Group0, THResNet101Group2, THResNet101Group1, THResNet50Group30, THResNet50Group31, THResNet50Group32
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












def train(layer, logger, shapes, args, e, data_size, trainloader):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(layer.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    train_loss = 0
    correct = 0
    total = 0
    criterion.cuda()

    layer.train()
    batch_idx = 0
    data_iter = iter(trainloader)
    back_flag = False
    outputs_queue = Q.Queue(args.buffer_size)
    while True:
        if dist.get_rank() == 0:
            if not back_flag:
                try:
                    inputs, targets = next(data_iter)
                    inputs = inputs.cuda()
                    outputs = layer(inputs)
                    outputs_queue.put(outputs)
                    send_opt = dist.isend(tensor=outputs.cpu(), dst=1)
                    send_opt.wait()
                except StopIteration as stop_e:
                    send_opt = dist.isend(tensor=torch.zeros(0), dst=1)
                    send_opt.wait()
                    ###
                    while not outputs_queue.empty():
                        try:
                            grad_recv = torch.zeros(shapes[0])
                            dist.recv(tensor=grad_recv, src=1)
                        except RuntimeError as error:
                            pass
                        grad_recv = grad_recv.cuda(0)
                        try:
                            loss = outputs_queue.get(block=True, timeout=4)
                            loss = loss.cuda(0)
                        except Empty:
                            print("empty........")
                            break
                        optimizer.zero_grad()
                        loss.backward(grad_recv)
                        optimizer.step()
                    ###
                    time.sleep(1)
                    e.set()
                    break
                if (batch_idx + 1) % 3 == 0:
                    back_flag = True
            else:
                grad_recv = torch.zeros(shapes[0])
                dist.recv(tensor=grad_recv, src=1)
                grad_recv = grad_recv.cuda(0)
                try:
                    loss = outputs_queue.get(block=True, timeout=4)
                    loss = loss.cuda(0)
                except Empty:
                    print("empty........")
                    break
                optimizer.zero_grad()
                loss.backward(grad_recv)
                optimizer.step()
                if (batch_idx + 1) % 3 == 0:
                    back_flag = False
            batch_idx += 1

        elif dist.get_rank() == 1:
            if not back_flag:
                try:
                    rec_val = torch.zeros(shapes[0])
                    dist.recv(tensor=rec_val, src=0)
                    rec_val = rec_val.cuda()
                    rec_val.requires_grad_()
                    outputs = layer(rec_val)
                    outputs_queue.put([rec_val, outputs])
                    send_opt = dist.isend(tensor=outputs.cpu(), dst=2)
                    send_opt.wait()
                except RuntimeError as error:
                    while not outputs_queue.empty():
                        grad_recv = torch.zeros(shapes[1])
                        dist.recv(tensor=grad_recv, src=2)
                        grad_recv = grad_recv.cuda(0)
                        try:
                            inputs, outputs = outputs_queue.get(block=True, timeout=4)
                        except Empty:
                            print("empty........")
                            break
                        inputs.requires_grad_()

                        optimizer.zero_grad()
                        outputs.backward(grad_recv)
                        optimizer.step()
                    e.wait()
                    break
                if (batch_idx + 1) % 3 == 0:
                    back_flag = True
            else:
                grad_recv = torch.zeros(shapes[1])
                dist.recv(tensor=grad_recv, src=2)
                grad_recv = grad_recv.cuda(0)
                try:
                    inputs, outputs = outputs_queue.get(block=True, timeout=4)
                except Empty:
                    print("empty........")
                    break
                inputs.requires_grad_()

                optimizer.zero_grad()
                outputs.backward(grad_recv)
                optimizer.step()
                send_opt = dist.isend(tensor=inputs.grad.cpu(), dst=0)
                sned_opt.wait()
                if (batch_idx + 1) % 3 == 0:
                    back_flag = False
            batch_idx += 1
        elif dist.get_rank() == 2:
            rec_val = torch.zeros(shapes[1])
            dist.recv(tensor=rec_val, src=1)
            for batch_idx, (_, targets) in enumerate(trainloader):
                rec_val = rec_val.cuda(0)
                rec_val.requires_grad_()
                outputs = layer(rec_val)
                # start to backward....
                targets = targets.cuda(0)
                if (batch_idx + 1) % 3 == 0:
                    loss = criterion(outputs, targets)
                    outputs_queue.put([loss, rec_val])
                    count = 0
                    while count < 3:

                        loss, rec_val = outputs_queue.get()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                        progress_bar(batch_idx, data_size, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                        logger.error("train:" + str(train_loss / (batch_idx + 1)))
                        acc_str = "tacc: %.3f" % (100. * correct / total,)
                        logger.error(acc_str)
                        send_opt = dist.isend(tensor=rec_val.grad.cpu(), dst=1)
                        send_opt.wait()
                        count += 1
                else:
                    loss = criterion(outputs, targets)
                    outputs_queue.put([loss, rec_val])
                    try:
                        rec_val = torch.zeros(shapes[1])
                        dist.recv(tensor=rec_val, src=1)
                    except RuntimeError as error:
                        while not outputs_queue.empty():
                            loss, rec_val = outputs_queue.get()
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item()
                            _, predicted = outputs.max(1)
                            total += targets.size(0)
                            correct += predicted.eq(targets).sum().item()
                            progress_bar(batch_idx, data_size, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                            logger.error("train:" + str(train_loss / (batch_idx + 1)))
                            acc_str = "tacc: %.3f" % (100. * correct / total,)
                            logger.error(acc_str)
                            send_opt = dist.isend(tensor=rec_val.grad.cpu(), dst=1)
                            send_opt.wait()
                        e.wait()
                        break











def eval(layer, logger, e, save_event, data_size, testloader):
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
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
                rec_val = torch.zeros([100, 480, 16, 16])  # difference model has difference shape
                dist.recv(tensor=rec_val, src=0)
                print("after recv....")
                outputs = layer(rec_val.cuda())
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
                rec_val = torch.zeros([100,  832, 8, 8])
                dist.recv(tensor=rec_val, src=1)
                outputs = layer(rec_val.cuda(0))
                targets = targets.cuda()
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
    logger = logging.getLogger(args.model + '-gpipe3-rank-' + str(dist.get_rank()))
    file_handler = logging.FileHandler(args.model + '-gpipe3-rank-' + str(dist.get_rank()) + '.log')
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
        time.sleep(1)
        print('Eval epoch: %d' % epoch)
        eval(layer, logger, epoch_event, save_event, test_size, testloader)
        epoch_event.clear()
        if save_event.is_set():
            print('Saving..')
            state = {
                'net': layer.state_dict(),
                'acc': best_acc,
                'epoch': 0,
            }
            torch.save(state, './checkpoint/' + args.model + '-gpipe3-rank-' + str(r) + '_ckpt.t7')
        time.sleep(1)
    if r == 0 or r == 1:
        global_event.wait()
    elif r == 1:
        global_event.set()





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
    node_cfg_0 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M']
    node_cfg_1 = [512, 512, 512, 512, 'M']
    node_cfg_2 = [512, 512, 512, 512, 'M']

    #vgg19
    #shapes = [[args.batch_size, 256, 4, 4], [args.batch_size, 512, 2, 2]]
    #res101
    #shapes = [[args.batch_size, 512, 16, 16], [args.batch_size, 1024, 8, 8]]
    #googlenet
    shapes = [[args.batch_size, 480, 16, 16], [args.batch_size, 832, 8, 8]]
    #res50
    #shapes = [[args.batch_size, 1024, 8, 8], [args.batch_size, 2048, 4, 4]]

    if args.rank == 0:
        #layer = THResNet101Group0()
        layer = GoogleNetGroup0()
        #layer = VggLayer(node_cfg_0)
        #layer = THDPNGroup0()
        #layer = THResNet50Group30()
        ## big model do not use
        layer.cuda()
    elif args.rank == 1:
        #layer = THResNet101Group1()
        layer = GoogleNetGroup1()
        #layer = VggLayer(node_cfg_1, node_cfg_0[-1] if node_cfg_0[-1] != 'M' else node_cfg_0[-2])
        #layer = THDPNGroup1()
        #layer = THResNet50Group31()
        layer.cuda()
    elif args.rank == 2:
        #layer = THResNet101Group2()
        layer = GoogleNetGroup2()
        #layer = VggLayer(node_cfg_2, node_cfg_1[-1] if node_cfg_1[-1] != 'M' else node_cfg_1[-2], last_flag=True)
        #layer = THDPNGroup2()
        #layer = THResNet50Group32()
        layer.cuda()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #layer.share_memory()
    cudnn.benchmark = True

    best_acc = 0.0
    start_epoch = 0
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/' + args.model + '-gpipe3-rank-' + str(args.rank) + '_ckpt.t7')
        layer.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print("best_acc: " + str(best_acc))
        print("start_epoch: " + str(start_epoch))

    print("init process-" + str(args.rank) + "....")
    dist.init_process_group(backend='tcp', init_method=args.path, world_size=args.size, rank=args.rank)

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
    run(start_epoch, layer, shapes, args, targets_queue, global_event, epoch_event, save_event, train_size, test_size, trainloader, testloader)