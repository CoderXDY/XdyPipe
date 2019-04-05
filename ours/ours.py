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
from model.res import THResNetGroup0, THResNetGroup1
from model.vgg_module import VggLayer
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
 pipeline ResNet script for Tianhe-2  with gpu cluster

"""




############################################
#top=n
############################################

def sparse2(tensor, k, half=True, residual=None):

    array_tensor = tensor.view(-1)
    if residual is None:
        residual = torch.zeros(array_tensor.size(), device=torch.device('cuda:1'))
    array_tensor.add_(residual)
    threshold = array_tensor.topk(int(array_tensor.nelement()*k) if int(array_tensor.nelement()*k) != 0 else 1)[0][-1]
    residual = torch.where(abs(array_tensor) < threshold, array_tensor, torch.zeros(array_tensor.size(), device=torch.device('cuda:1')))
    array_tensor[abs(array_tensor) < threshold] = 0.
    indexs = array_tensor.nonzero().t()
    values = array_tensor[indexs[0]]
    sparse_tensor = torch.cat([indexs[0].float(), values])
    if half:
        sparse_tensor = sparse_tensor.half()

    return sparse_tensor, residual

def dense(tensor, shape):
    if tensor.type() != 'torch.FloatTensor':
        tensor = tensor.float()
    half_size = int(len(tensor) / 2)
    indexs = tensor[: half_size].view(1, half_size).long()
    values = tensor[half_size:]
    length = 1
    for i in range(len(shape)):
        length *= shape[i]
    sparse_tensor = torch.sparse.FloatTensor(indexs, values, torch.Size([length]))
    return sparse_tensor.to_dense().view(shape)

################################################
##quantize
#################################################

def quantize(input, num_bits=8, char=False, residual=None):
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.
    scale = qmax - qmin
    input = torch.round(input.mul(scale).mul(1000))
    if char:
        input = input.char()
    return input

    #b = torch.abs(a)
    #c = torch.max(b)
    #torch.round(torch.abs(a).mul(255).div(c))

def dequantize(input, shape, num_bits=8):
    if input.type() != 'torch.FloatTensor':
        input = input.float()
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.
    scale = qmax - qmin
    input.div_(1000 * scale)
    return input









def train(layer, logger, args, grad_queue, targets_queue, e, data_size, trainloader, start_event):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(layer.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    optimizer.zero_grad()
    layer.train()

    def backward():
        start_event.wait()
        batch_idx = 0
        while True:
            try:
                grad = grad_queue.get(block=True, timeout=1)
                #quantize_package = grad_queue.get(block=True, timeout=1)
                #grad = dequantize(quantize_package, [args.batch_size, 256, 4, 4])
                #grad = grad.cuda()
                #grad = torch.from_numpy(grad)
                #grad = dense(grad, [args.batch_size, 256, 4, 4]).cuda(0)
                #grad = torch.from_numpy(grad).cuda(0).float()
                grad = torch.from_numpy(grad.astype(np.float32)).cuda(0)
                #grad = dequantize(grad, [args.batch_size, 256, 4, 4])
                grad = dequantize(grad, [args.batch_size, 256, 4, 4])

            except Empty as empty:
                print("backward empty.....")
                break
            loss = outputs_queue.get(block=False)
            loss.backward(grad)
            if batch_idx % args.buffer_size == 0:
                optimizer.step()
                optimizer.zero_grad()
            batch_idx += 1


    if dist.get_rank() == 0:
        criterion.cuda(0)
        outputs_queue = ThreadQueue(args.buffer_size)
        back_process = Process(target=backward)
        back_process.start()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            print("batch: " + str(batch_idx))
            inputs, targets = inputs.cuda(0), targets
            outputs = layer(inputs)
            send_opt = dist.isend(tensor=quantize(outputs,char=True).cpu(), dst=1)
            # if batch_idx < 30:
            send_opt.wait()
            targets_queue.put(targets.numpy())
            outputs_queue.put(outputs)
            print("send....")
        send_opt = dist.isend(tensor=torch.zeros(0), dst=1)
        send_opt.wait()
        back_process.join()
        e.set()
    elif dist.get_rank() == 1:
        batch_idx = 0
        train_loss = 0
        correct = 0
        total = 0
        criterion.cuda(1)
        residual = None
        while True:
            try:
                rec_val = torch.zeros([args.batch_size, 256, 4, 4], dtype=torch.int8)
                dist.recv(tensor=rec_val, src=0)
            except RuntimeError as error:
                e.wait()
                break
            rec_val = rec_val.cuda(1)
            rec_val.requires_grad_()
            outputs = layer(rec_val)
            targets = targets_queue.get(block=True, timeout=2)
            targets = torch.from_numpy(targets).cuda(1)
            loss = criterion(outputs, targets)
            loss.backward()
            #spare_grad, residual = sparse2(rec_val.grad, 0.01, True, residual)
            #grad_queue.put(spare_grad.cpu().numpy())
            #print('before grad put')
            #grad_queue.put(rec_val.grad.cpu().half().numpy())
            #print('after grad put')
            #quantize_grad = quantize(rec_val.grad, num_bits=args.bit, half=True)
            #grad_queue.put(quantize_grad.cpu().numpy())
            #quantize_package = quantize(rec_val.grad, num_bits=args.bit, byte=True)
            #grad_queue.put(quantize_package)
            quantize_grad = quantize(rec_val.grad)
            grad_queue.put(quantize_grad.cpu().numpy().astype(np.int8))
            if batch_idx == 0:
                start_event.set()
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
            #if batch_idx % 10 == 0:
            logger.error("train:" + str(train_loss / (batch_idx + 1)))

            batch_idx += 1
            acc_str = "tacc: %.3f" % (100. * correct / total,)
        logger.error(acc_str)

def eval(layer, logger, args, targets_queue, e, save_event, data_size, testloader):
    criterion = nn.CrossEntropyLoss()
    criterion.cuda(1)
    layer.eval()

    with torch.no_grad():
        if dist.get_rank() == 0:
            for batch_idx, (inputs, targets) in enumerate(testloader):
                print('batch_idx: ' + str(batch_idx))
                inputs, targets = inputs.cuda(0), targets
                outputs = layer(inputs)
                targets_queue.put(targets.numpy())
                send_opt = dist.isend(tensor=outputs.cpu(), dst=1)
                send_opt.wait()
            send_opt = dist.isend(tensor=torch.zeros(0), dst=1)
            send_opt.wait()
            e.wait()
        elif dist.get_rank() == 1:
            batch_idx = 0
            test_loss = 0
            correct = 0
            total = 0
            save_event.clear()
            global best_acc
            while True:
                try:
                    rec_val = torch.zeros([100, 256, 4, 4])
                    dist.recv(tensor=rec_val, src=0)
                except RuntimeError as error:
                    print("done....")
                    acc = 100. * correct / total
                    if acc > best_acc:
                        best_acc = acc
                        save_event.set()
                    e.set()
                    break
                outputs = layer(rec_val.cuda(1))
                targets = targets_queue.get(block=True, timeout=2)
                targets = torch.from_numpy(targets).cuda(1)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, data_size, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                #if batch_idx % 10 == 0:
                logger.error("eval:" + str(test_loss / (batch_idx + 1)))
                batch_idx += 1
            acc_str = "eacc: %.3f" % (100. * correct / total,)
            logger.error(acc_str)



def run(start_epoch, layer, args, grad_queue, targets_queue, global_event, epoch_event, save_event, train_size, test_size, trainloader, testloader, start_event):
    logger = logging.getLogger(args.model + 'ours-rank-' + str(dist.get_rank()))
    file_handler = logging.FileHandler(args.model + '-ours-rank-' + str(dist.get_rank()) + '.log')
    file_handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter(fmt='%(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    epoch_num = 200
    global best_acc
    r = dist.get_rank()
    for epoch in range(start_epoch, start_epoch + epoch_num):
        print('Training epoch: %d' % epoch)
        train(layer, logger, args, grad_queue, targets_queue, epoch_event, train_size, trainloader, start_event)
        epoch_event.clear()
        start_event.clear()
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
            torch.save(state, './checkpoint/'+ args.model + '-ours-rank-' + str(r) + '_ckpt.t7')
    if r == 0:
        global_event.wait()
    elif r == 1:
        global_event.set()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', help='the ip of master', default='89.72.2.41')
    parser.add_argument('-size', type=int, help='input the sum of node', default=2)
    parser.add_argument('-path', help='the path fo share file system',
                        default='file:///WORK/sysu_wgwu_4/xpipe/ours/temp')
    parser.add_argument('-rank', type=int, help='the rank of process')
    parser.add_argument('-layer_type', type=int, help='type of layer: input:0, block:1, output:2')
    parser.add_argument('-batch_size', type=int, help='size of batch', default=64)
    parser.add_argument('-data_worker', type=int, help='the number of dataloader worker', default=2)
    parser.add_argument('-epoch', type=int)
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('-model', help='the path fo share file system')
    parser.add_argument('-buffer_size', type=int, help='size of batch', default=4)
    parser.add_argument('-bit', type=int, help='the rank of process', default=8)
    args = parser.parse_args()
    print("ip: " + args.ip)
    print("size: " + str(args.size))
    print("path: " + args.path)
    print("rank: " + str(args.rank))
    print("layer_type: " + str(args.layer_type))
    print("batch_size: " + str(args.batch_size))
    print("data_worker: " + str(args.data_worker))
    print("model: " + str(args.model))

    #torch.manual_seed(1)

    bm.register('get_epoch_event')
    bm.register('get_global_event')
    bm.register('get_grad_queue')
    bm.register('get_targets_queue')
    bm.register('get_save_event')
    bm.register('get_backward_event')
    bm.register('get_start_thread_event')
    m = bm(address=(args.ip, 5002), authkey=b'xpipe')
    m.connect()
    global_event = m.get_global_event()
    epoch_event = m.get_epoch_event()
    grad_queue = m.get_grad_queue()
    targets_queue = m.get_targets_queue()
    save_event = m.get_save_event()
    start_event = m.get_start_thread_event()


    """
        difference model
        """
    node_cfg_0 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M']
    node_cfg_1 = [512, 512, 512, 'M', 512, 512, 512, 'M']

    if args.rank == 0:
        layer = VggLayer(node_cfg_0)
        layer.cuda(0)
    elif args.rank == 1:
        layer = VggLayer(node_cfg_1, node_cfg_0[-1] if node_cfg_0[-1] != 'M' else node_cfg_0[-2], True)
        layer.cuda(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    layer.share_memory()
    cudnn.benchmark = True

    best_acc = 0.0
    start_epoch = 0
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/' + args.model + '-ours-rank-' + str(args.rank) + '_ckpt.t7')
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

    
    run(start_epoch, layer, args, grad_queue, targets_queue, global_event, epoch_event, save_event, train_size, test_size, trainloader, testloader, start_event)

