import torch
import torch.distributed as dist
from torch import nn as nn
import argparse
#from torch.multiprocessing import Process, Queue, Value, Event
#from multiprocessing.managers import BaseManager as bm
from multiprocessing.dummy import Process, Queue
import torch.nn.functional as F
import torch.optim as optim
import logging
import time
import torchvision
import torchvision.transforms as transforms
from utils import progress_bar
import traceback
from queue import Empty, Full
import os
import psutil
import gc
from resnet import ResNet18
from resnet152_dist import ResNet50
from resnet_pipe_model import ResPipeNet18, ResPipeNet50, THResPipeNet18, THResPipeNet50
import torch.backends.cudnn as cudnn
from visdom import Visdom


parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('-file', help='the filename of log')
parser.add_argument('-buffer_size', type=int, help='the size of buffer queue caching the batch data', default=3)
parser.add_argument('-batch_size', type=int, help='batch_size', default=128)
parser.add_argument('-wait', type=int, help='wait to start thread', default=2)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('-mq', type=int, help='queue size of model', default=1)
parser.add_argument('-mw', type=int, help='wait time of model', default=0)
parser.add_argument('-count', type=int, help='first forward', default=1)

args = parser.parse_args()


logger = logging.getLogger(args.file)
file_handler = logging.FileHandler(args.file + '.log')
file_handler.setLevel(level=logging.DEBUG)
formatter = logging.Formatter(fmt='%(message)s', datefmt='%Y/%m/%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
epoch_loss = 0.0
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')







cudnn.benchmark = True



net = ResPipeNet50(args.mq, args.mw)







if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print("best_acc: " + str(best_acc))
    print("start_epoch: " + str(start_epoch))



criterion = nn.CrossEntropyLoss()
criterion.cuda(1)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


output_queue = Queue(args.buffer_size)



def train(epoch):

    print('Epoch: %d' % epoch)

    def backward():

        time.sleep(2)

        batch_idx = 0
        train_loss = 0
        correct = 0
        total = 0
        global epoch_loss

        while True:
            optimizer.zero_grad()
            try:
                outputs, targets = output_queue.get(block=True, timeout=args.wait)
            except Empty as e:
                print("done.....")
                epoch_loss = (train_loss / (batch_idx + 1))
                break
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            batch_idx += 1

    net.train()

    start_flag = True
    first_count = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(0), targets.to(1)
        outputs = net(inputs)
        if first_count < args.count:
            first_count += 1
            continue
        output_queue.put([outputs, targets])
        if start_flag and output_queue.qsize() > args.wait: #2
            start_flag = False
            back_process = Process(target=backward)
            back_process.start()


    back_process.join()


def test(epoch):
    global best_acc
    global epoch_loss
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(0), targets.to(1)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        epoch_loss = test_loss/(batch_idx+1)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc




for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    logger.error("--train_epoch:" + str(epoch) + "--loss:" + str(epoch_loss))
    test(epoch)
    logger.error("--test_epoch:" + str(epoch) + "--loss:" + str(epoch_loss))