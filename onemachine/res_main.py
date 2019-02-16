import torch
import torch.distributed as dist
from torch import nn as nn
import argparse
#from torch.multiprocessing import Process, Queue, Value, Event
#from multiprocessing.managers import BaseManager as bm
import torch.nn.functional as F
import torch.optim as optim
import logging
import time
import torchvision
import torchvision.transforms as transforms
from utils import progress_bar
import traceback
#from queue import Empty, Full
import os
import psutil
import gc



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    torch.manual_seed(1)
    net = net.to(device)
    net.share_memory()
    #torch.multiprocessing.set_start_method("spawn")
    #cudnn.benchmark = True

    f_p = Process(target=forward_train(), args=(net, ))
    f_p.start()
    b_p = Process(target=backward_train(), args=(net, ))
    b_p.start()
    f_p.join()
    b_p.join()