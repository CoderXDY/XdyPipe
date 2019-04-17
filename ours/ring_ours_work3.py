import torch
import torch.distributed as dist
from torch import nn as nn
import argparse

from torch.multiprocessing import Queue, Event
from multiprocessing.managers import BaseManager as bm
from multiprocessing.dummy import Process
from multiprocessing.dummy import Queue as ThreadQueue
from multiprocessing.dummy import Semaphore
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


def get_left_right(rank, size):

    return left, right


def transfer(rank, send_buf, shape):
    left, right = get_left_right(rank)
    send_opt = dist.isend(tensor=send_buf, dst=right)

    try:
        recv_buf = torch.zeros(shape, dtype=torch.int8)
        dist.recv(tensor=recv_buf, src=left)
    except RuntimeError as error:
        print("rank:" + str(rank) + " start to end.....")
        return None
    send_opt.wait()
    return recv_buf


def parallel(layer, trainloader, rank, targets_queue, e):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(layer.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    optimizer.zero_grad()
    layer.train()
    criterion.cuda()
    batch_idx = 0
    if rank == 0:
        outputs_queue = ThreadQueue(args.buffer_size)
        back_process = Process(target=backward_rank0, args=())
        back_process.start()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets
            targets_queue.put(targets.numpy())
            outputs = layer(inputs)
            outputs_queue.put(outputs)
            recv_buf = transfer(rank, outputs, [])
        transfer(rank, torch.zeros(0),[])
        e.wait()
        back_process.join()

    elif rank == 1:
        rec_val = torch.randn()
        while True:
            if rec_val == None:
                e.wait()
                break
            else:
                rec_val = dq_act(rec_val)
                rec_val = rec_val.cuda(0)
                rec_val.requires_grad_()
                outputs = layer(rec_val)
                outputs_queue.put([rec_val, outputs])
                rec_val = transfer(rank, outputs, [])

    elif rank == 2:
        rec_val = torch.randn()
        count = 0
        while True:
            if rec_val == None:
                e.wait()
                break
            else:
                rec_val = dq_act(rec_val)
                rec_val = rec_val.cuda(0)
                rec_val.requires_grad_()
                outputs = layer(rec_val)

                if count < 5:
                    targets = targets_queue.get(block=True, timeout=2)
                    targets = torch.from_numpy(targets).cuda()
                    loss = criterion(outputs, targets)
                    loss.backward()
                    quantize_grad = quantize(rec_val.grad, char=True).cpu()
                    rec_val = transfer(rank, quantize_grad, [])
                    optimizer.zero_grad()
                    count += 1
                    continue

                # start to backward....
                targets = targets_queue.get(block=True, timeout=2)
                targets = torch.from_numpy(targets).cuda()
                loss = criterion(outputs, targets)
                loss.backward()

                if batch_idx % args.buffer_size == 0:
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
                # if batch_idx % 10 == 0:
                logger.error("train:" + str(train_loss / (batch_idx + 1)))
                acc_str = "tacc: %.3f" % (100. * correct / total,)
                logger.error(acc_str)
                quantize_grad = quantize(rec_val.grad, char=True).cpu()
                rec_val = transfer(rank, quantize_grad, [])

    def backward_rank1():
        grad_recv1 = torch.zeros(0)
        count = 0
        while True:
            if grad_recv1 == None:
                e.wait()
                break
            grad_recv1 = dequantize(grad_recv1.cuda(0).float())
            inputs, outputs = outputs_queue.get(block=False)
            inputs.requires_grad_()
            outputs.backward(grad_recv1)
            if count < 4:
                inputs_grad = quantize(inputs.grad, char=True).cpu()
                transfer(rank, inputs_grad, [])
                optimizer.zero_grad()
                count += 1
                continue
            if batch_idx % args.buffer_size == 0:
                optimizer.step()
                optimizer.zero_grad()
            inputs_grad = quantize(inputs.grad, char=True).cpu()
            transfer(rank, inputs_grad, [])

    def backward_rank0():
        grad_recv = torch.zeros(0)
        count = 0
        while True:
            if grad_recv == None:
                e.set()
                break
            grad_recv = dequantize(grad_recv.cuda(0).float())
            loss = outputs_queue.get(block=False)
            loss.backward(grad_recv)
            if count < 5:
                grad_recv = transfer(rank, torch.zeros(0), [])
                optimizer.zero_grad()
                count += 1
                continue

            if batch_idx % args.buffer_size == 0:
                optimizer.step()
                optimizer.zero_grad()











    if rank == 0:

    elif rank == 1:
        back_process = Process(target=backward_rank1, args=())
        back_process.start()


    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if rank == 0:
            inputs, targets = inputs.cuda(), targets
            targets_queue.put(targets.numpy())
            outputs = layer(inputs)
            outputs_queue.put(outputs)
            recv_buf = transfer(rank, outputs, [])
        elif rank == 1:
            if recv_val == None:
                e.wait()
                break
            else:
                rec_val = dq_act(rec_val)
                rec_val = rec_val.cuda(0)
                rec_val.requires_grad_()
                outputs = layer(rec_val)
                outputs_queue.put([rec_val, outputs])
                rec_val = transfer(rank, outputs, [])
        elif rank == 2:
            if recv_val == None:
                e.wait()
                break
            else:
                rec_val = dq_act(rec_val)
                rec_val = rec_val.cuda(0)
                rec_val.requires_grad_()
                outputs = layer(rec_val)

                # start to backward....
                targets = targets_queue.get(block=True, timeout=2)
                targets = torch.from_numpy(targets).cuda()
                loss = criterion(outputs, targets)
                loss.backward()

                if batch_idx % args.buffer_size == 0:
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
                # if batch_idx % 10 == 0:
                logger.error("train:" + str(train_loss / (batch_idx + 1)))
                acc_str = "tacc: %.3f" % (100. * correct / total,)
                logger.error(acc_str)
                quantize_grad = quantize(rec_val.grad, char=True).cpu()
                rec_val = transfer(rank, quantize_grad, [])

    def backward_rank1():
        while True:
            if grad_recv1 == None:
                e.wait()
                break
            grad_recv1 = dequantize(grad_recv1.cuda(0).float())
            inputs, outputs = outputs_queue.get(block=False)
            inputs.requires_grad_()
            outputs.backward(grad_recv1)
            if batch_idx % args.buffer_size == 0:
                optimizer.step()
                optimizer.zero_grad()
            inputs_grad = quantize(inputs.grad, char=True).cpu()
            transfer(rank, inputs_grad, [])

    def backward_rank0():
        while True:
            if grad_recv == None:
                e.wait()
                break
            grad_recv = dequantize(grad_recv.cuda(0).float())
            loss = outputs_queue.get(block=False)
            loss.backward(grad_recv)
            if batch_idx % args.buffer_size == 0:
                optimizer.step()
                optimizer.zero_grad()





def train():
    parallel()
