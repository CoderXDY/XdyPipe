import torch
import torch.distributed as dist
from torch import nn as nn
import argparse

from torch.multiprocessing import Queue, Event
from multiprocessing.managers import BaseManager as bm
from multiprocessing.dummy import Process
from multiprocessing.dummy import Queue as ThreadQueue
import torch.nn.functional as F
import torch.optim as optim
import logging
import time
from model.res import THResNet101Group0, THResNet101Group2, THResNet101Group1
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
"""
 
 pipedream 

"""



def pipe_dream(layer, logger, args, backward_event, targets_queue, e, data_size, trainloader):

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(layer.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    layer.train()

    if dist.get_rank() == 0:
        output_queue = ThreadQueue(3)
        data_iter = iter(trainloader)
        batch_idx = 0
        while True:
            try:
                if output_queue.qsize() == 3:
                    #backward_event.wait()
                    optimizer.zero_grad()
                    grad = torch.zeros([args.batch_size, 256, 4, 4])
                    dist.recv(tensor=grad, src=1)
                    print("qsize after recv")
                    outputs = output_queue.get()
                    outputs.backward(grad.cuda())
                    optimizer.step()
                    #backward_event.clear()
                    continue
                else:
                    inputs, targets = next(data_iter)
                    inputs = inputs.cuda()
                    targets_queue.put(targets.numpy(), block=False)
                    print("after target...")
                    outputs = layer(inputs)
                    send_opt = dist.isend(tensor=outputs.cpu(), dst=1)
                    send_opt.wait()
                    print("after wait....")
                    output_queue.put(outputs)
                    print("after output")
                    batch_idx += 1
            except StopIteration as stop_e:
                send_opt = dist.isend(tensor=torch.zeros(0), dst=1)
                send_opt.wait()
                # while output_queue.qsize() > 0:
                #     print("start stop......")
                #     #backward_event.wait()
                #     optimizer.zero_grad()
                #     grad = torch.zeros([args.batch_size, 256, 4, 4])
                #     dist.recv(tensor=grad, src=1)
                #     print("stop recv...")
                #     outputs = output_queue.get()
                #     print("stop get")
                #     outputs.backward(grad.cuda())
                #     optimizer.step()
                #     #backward_event.clear()
                #time.sleep(1)
                #e.set()
                e.wait()
                break

    elif dist.get_rank() == 1:
        batch_idx = 0
        output_queue = ThreadQueue(3)
        while True:
            if output_queue.qsize() == 3:
               #backward_event.wait()
                optimizer.zero_grad()
                grad = torch.zeros([args.batch_size, 512, 2, 2])
                dist.recv(tensor=grad, src=2)
                print("qsize after recv")
                outputs, inputs = output_queue.get()
                inputs.requires_grad_()
                outputs.backward(grad.cuda())
                optimizer.step()
                send_opt = dist.isend(tensor=inputs.grad.cpu(), dst=0)
                send_opt.wait()
                print("qsize after wait...")
                #backward_event.clear()
                continue
            else:
                try:
                    print("start....")
                    rec_val = torch.zeros([args.batch_size, 256, 4, 4])
                    dist.recv(tensor=rec_val, src=0)
                    print("after recv")
                except RuntimeError as error:
                    print("no data from rank 0 ....")
                    send_opt = dist.isend(tensor=torch.zeros(0), dst=2)
                    send_opt.wait()
                    # while output_queue.qsize() > 0:
                    #     backward_event.wait()
                    #     optimizer.zero_grad()
                    #     grad = torch.zeros([args.batch_size, 512, 2, 2])
                    #     dist.recv(tensor=grad, src=2)
                    #     outputs, inputs = output_queue.get()
                    #     inputs.requires_grad_()
                    #     outputs.backward(grad.cuda())
                    #     optimizer.step()
                    #     send_opt = dist.isend(tensor=inputs.grad.cpu(), dst=0)
                    #     send_opt.wait()
                    #     backward_event.clear()
                    e.wait()
                    break

                rec_val = rec_val.cuda()
                rec_val.requires_grad_()
                outputs = layer(rec_val)
                print("after output")
                output_queue.put([outputs, rec_val])
                send_opt = dist.isend(tensor=outputs.cpu(), dst=2)
                #send_opt.wait()
                print("after wait...")
                batch_idx += 1



    elif dist.get_rank() == 2:
        batch_idx = 0
        train_loss = 0
        correct = 0
        total = 0
        while True:
            try:
                #print("start.....")
                rec_val = torch.zeros([args.batch_size, 512, 2, 2])
                dist.recv(tensor=rec_val, src=1)
                #print("after recv......")
            except RuntimeError as error:
                print("runtime........................")
                time.sleep(1)
                #e.wait()
                e.set()
                break
            rec_val = rec_val.cuda()
            rec_val.requires_grad_()
            optimizer.zero_grad()
            outputs = layer(rec_val)
            #print("after output")
            targets = targets_queue.get(block=True, timeout=2)
            targets = torch.from_numpy(targets).cuda()
            #print("after target")
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, data_size, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            # if not backward_event.is_set():
            #     backward_event.set()
            send_opt = dist.isend(tensor=rec_val.grad.cpu(), dst=1)
            #send_opt.wait()
            #print("after send")
            #if batch_idx % 10 == 0:
            logger.error("train:" + str(train_loss / (batch_idx + 1)))
            acc_str = "tacc: %.3f" % (100. * correct / total,)
            logger.error(acc_str)
            batch_idx += 1




def eval(layer, logger, args, targets_queue, e, save_event, data_size, testloader):
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    layer.eval()

    with torch.no_grad():
        if dist.get_rank() == 0:
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.cuda(0), targets
                outputs = layer(inputs)
                targets_queue.put(targets.numpy())
                print("after put....")
                send_opt = dist.isend(tensor=outputs.cpu(), dst=1)
                send_opt.wait()
            send_opt = dist.isend(tensor=torch.zeros(0), dst=1)
            send_opt.wait()
            e.wait()
        elif dist.get_rank() == 1:
            batch_idx = 0
            while True:
                try:
                    rec_val = torch.zeros([100, 256, 4, 4])  # difference model has difference shape
                    dist.recv(tensor=rec_val, src=0)
                except RuntimeError as error:
                    send_opt = dist.isend(tensor=torch.zeros(0), dst=2)
                    send_opt.wait()
                    e.wait()
                    break
                outputs = layer(rec_val.cuda())
                send_opt = dist.isend(tensor=outputs.cpu(), dst=2)
                send_opt.wait()
                batch_idx += 1

        elif dist.get_rank() == 2:
            batch_idx = 0
            test_loss = 0
            correct = 0
            total = 0
            save_event.clear()
            global best_acc
            while True:
                try:
                    rec_val = torch.zeros([100, 512, 2, 2]) #difference model has difference shape
                    dist.recv(tensor=rec_val, src=1)
                except RuntimeError as error:
                    print("done....")
                    acc = 100. * correct / total
                    if acc > best_acc:
                        best_acc = acc
                        save_event.set()
                    e.set()
                    break
                outputs = layer(rec_val.cuda())
                targets = targets_queue.get(block=True, timeout=2)
                targets = torch.from_numpy(targets).cuda()
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, data_size, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                #if batch_idx % 10 == 0:
                logger.error("eval:" + str(test_loss / (batch_idx + 1)))
                acc_str = "eacc: %.3f" % (100. * correct / total,)
                logger.error(acc_str)
                batch_idx += 1


def run(start_epoch, layer, args, grad_queue, targets_queue, global_event, epoch_event, save_event, train_size, test_size, trainloader, testloader, backward_event=None):
    logger = logging.getLogger(args.model + '-rank-' + str(dist.get_rank()))
    file_handler = logging.FileHandler(args.model + '-pipedream-rank-' + str(dist.get_rank()) + '.log')
    file_handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter(fmt='%(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    epoch_num = 200
    global best_acc
    r = dist.get_rank()
    for epoch in range(start_epoch, start_epoch + epoch_num):
        print('Training epoch: %d' % epoch)
        pipe_dream(layer, logger, args, backward_event, targets_queue, epoch_event, train_size, trainloader)
        epoch_event.clear()
        time.sleep(1)
        print('Eval epoch: %d' % epoch)
        eval(layer, logger, args, targets_queue, epoch_event, save_event, test_size, testloader)
        epoch_event.clear()
        if save_event.is_set():
            print('Saving..')
            state = {
                'net': layer.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
            }
            torch.save(state, './checkpoint/' + args.model + '-pipedream-rank-' + str(r) + '_ckpt.t7')
        time.sleep(1)
    if r == 0 or r == 1:
        global_event.wait()
    elif r == 2:
        global_event.set()



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
    parser.add_argument('-mode', type=int, help='the rank of process', default=0)
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
    bm.register('get_targets_queue')
    bm.register('get_save_event')
    bm.register('get_backward_event')
    m = bm(address=(args.ip,args.port), authkey=b'xpipe')
    m.connect()
    global_event = m.get_global_event()
    epoch_event = m.get_epoch_event()
    grad_queue = m.get_grad_queue()
    targets_queue = m.get_targets_queue()
    save_event = m.get_save_event()
    backward_event = None
    if args.mode != 0:
        backward_event = m.get_backward_event()

    node_cfg_0 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M']
    node_cfg_1 = [512, 512, 512, 512, 'M']
    node_cfg_2 = [512, 512, 512, 512, 'M']

    if args.rank == 0:
        #layer = THResNet101Group0()
        layer = VggLayer(node_cfg_0)
        layer.cuda()
    elif args.rank == 1:
        #layer = THResNet101Group1()
        layer = VggLayer(node_cfg_1, node_cfg_0[-1] if node_cfg_0[-1] != 'M' else node_cfg_0[-2])
        layer.cuda()
    elif args.rank == 2:
        #layer = THResNet101Group2()
        layer = VggLayer(node_cfg_2, node_cfg_1[-1] if node_cfg_1[-1] != 'M' else node_cfg_1[-2], last_flag=True)
        layer.cuda()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #layer.share_memory()
    cudnn.benchmark = True

    best_acc = 0.0
    start_epoch = 0
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/' + args.model + '-pipedream-rank-' + str(args.rank) + '_ckpt.t7')
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
    if args.rank == 1:
        trainloader = None
        testloader = None

    if args.epoch != 0:
        start_epoch = args.epoch
    run(start_epoch, layer, args, grad_queue, targets_queue, global_event, epoch_event, save_event, train_size, test_size, trainloader, testloader, backward_event)

