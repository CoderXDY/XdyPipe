import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
import time
import copy
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--dist-rank', default=0, type=int)
parser.add_argument('--dir-data-train', default='./data/tiny-imagenet-200/train', type=str)
parser.add_argument('--dir-data-val',default='./data/tiny-imagenet-200/val_temp',type=str)
parser.add_argument('--world-size', default=4, type=int)
parser.add_argument('--lr',default=0.1,type=float)
parser.add_argument('--epochs',default=300,type=int)
parser.add_argument('--file-name',default='ResNet18_Aji_4(300)',type=str)
parser.add_argument('--init-method',default='file:///WORK/sysu_wgwu_3/share_file_kuangdi',type=str)


Total_param_num = 0
Sparse_param_num = 0
criterion = nn.CrossEntropyLoss()

def trans_gradients(net):
    world_size = float(dist.get_world_size())
    global index
    global Total_param_num
    global Sparse_param_num
    Total_param_num = 0
    Sparse_param_num = 0


    Param_size = []
    Param_Total = torch.rand(0).cuda()
    for i, param in enumerate(net.parameters()) :
        Param_Temp = copy.deepcopy(param.grad.data)
        Param_size.append(Param_Temp.size())
        Param_Total = torch.cat((Param_Total, Param_Temp.view(-1)),0)

    # print (total_num)
    Total_param_num = Param_Total.size()[0]

    value = Param_Total[Param_Total != 0]

    Sparse_param_num = len(value)

    Temp_Param_Total = Param_Total.view(1,-1).cpu()


    if len(value) != 0 :
        index = torch.nonzero(torch.where(Temp_Param_Total!=0, torch.ones(Temp_Param_Total.size()), torch.zeros(Temp_Param_Total.size()) )).t().cuda()
    else :
        index = torch.LongTensor([]).cuda()


    global index_list
    global value_list
    global size_list

    # Get index and values size
    # selfsize
    size = torch.IntTensor( [len(value) if len(value) != 0 else 0] ).cuda()

    # initial size_list for gather whole group tensor of size
    size_list = [torch.IntTensor([0]).cuda() for j in range(args.world_size)]
    dist.all_gather(size_list, size)

    max_size = int(max(size_list[j][0] for j in range(args.world_size)))

    if max_size != 0:
        # Initial supplement_index and supplement_value
        supplement_index = torch.LongTensor([-1 for j in range(max_size - len(value) if len(value) != 0 else 0)] ).cuda()
        temp_index = torch.LongTensor(2,max_size).cuda()
        temp_index[0] = torch.cat((index[0], supplement_index), 0)
        temp_index[1] = torch.cat((index[1], supplement_index), 0)

        index = temp_index

        supplement_value = torch.FloatTensor([-1 for j in range(max_size - len(value) if len(value) != 0 else 0)]).cuda()
        value = torch.cat((value, supplement_value), 0)


        # Initial index_list for gather whole group tensor of index
        index_list = [torch.LongTensor([ [2 for k in range(max_size)] for j in range(2)]).cuda() for l in range(args.world_size)]
        dist.all_gather(index_list, index)


        # Initial value_list for gather whole group tensor of value
        value_list = [torch.FloatTensor([2 for k in range(max_size)]).cuda() for l in range(args.world_size)]
        dist.all_gather(value_list, value)


        # Wash the data of index and values
        Param_Merge = torch.zeros(Total_param_num).cuda().view(1,-1)
        for j in range(args.world_size) :
            index_list[j] = index_list[j][index_list[j] != -1].view(2,-1)
            value_list[j] = value_list[j][value_list[j] != -1]

            if len(index_list[j][0]) != 0 and len(value_list[j]) != 0:
                 Param_Merge += torch.sparse.FloatTensor(index_list[j], value_list[j], Param_Merge.size()).to_dense().cuda()

        # Update Parameter
        end = 0
        for i, param in enumerate(net.parameters()) :
            start = end
            end += param.grad.data.nelement()
            param.grad.data = (param.grad.data + Param_Merge[0][start:end].view(Param_size[i])) / world_size


def gradient_execute(net):
    threshold = 0.002
    paralist = []
    for param in net.parameters():
        temp = copy.deepcopy(param.grad.data)
        topn = torch.topk(abs(temp.view(1,-1)),int(temp.nelement()*0.01) if int(temp.nelement()*0.01) != 0 else 1)
        threshold = float(topn[0][0][len(topn[0][0])-1])
        temp[abs(temp) >= threshold] = 0
        param.grad.data[abs(param.grad.data) < threshold] = 0
        paralist.append(temp)

    trans_gradients(net)
    return paralist

# 获取数据
def run():

    #Model
    net = ResNet18()
    net = net.cuda()


    #Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    train_dataset = torchvision.datasets.ImageFolder(root=args.dir_data_train, transform=transform_train)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=(train_sampler is None), num_workers=2, pin_memory=True, sampler=train_sampler)
    test_dataset = torchvision.datasets.ImageFolder(root=args.dir_data_val, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    # Optimizer and scheduler of Training
    optimizer = optim.SGD(net.parameters(), lr=args.lr)


    #Training
    top1 = AverageMeter()
    top5 = AverageMeter()
    logs = []
    print("Training Start")
    for epoch in range(args.epochs):

        print("Training for epoch {}".format(epoch))
        net.train()
        train_sampler.set_epoch(epoch)
        for i, data in enumerate(train_loader,0):
            batch_start_time = time.time()
            x, label = data
            x, label = Variable(x).cuda(), Variable(label).cuda()

            # optimizer.zero_grad()
            output = net(x)
            loss = criterion(output, label)

            prec1 , prec5 = accuracy(output.data, label, topk=(1,5))
            top1.update(prec1[0], x.size(0))
            top5.update(prec5[0], x.size(0))


            loss.backward()
            paralist = gradient_execute(net)
            optimizer.step()
            for para1, para2 in zip(paralist, net.parameters()):
                para2.grad.data = para1



            log_obj = {
                'timestamp': datetime.now(),
                'iteration': i,
                'training_loss': loss.data.item(),
                'training_accuracy1': top1.avg.item()/100.0,
                'training_accuracy5':top5.avg.item()/100.0,
                'total_param': Total_param_num,
                'sparse_param': Sparse_param_num,
                'mini_batch_time': (time.time() - batch_start_time)
            }
            if i % 20 == 0:
                print("Timestamp: {timestamp} | "
                      "Iteration: {iteration:6} | "
                      "Loss: {training_loss:6.4f} | "
                      "Accuracy1: {training_accuracy1:6.4f} | "
                      "Accuracy5: {training_accuracy5:6.4f} | "
                      "Total_param: {total_param:6} | "
                      "Sparse_param: {sparse_param:6} | "
                      "Mini_Batch_Time: {mini_batch_time:6.4f} | ".format(**log_obj))

            logs.append(log_obj)

        if True:
            logs[-1]['test_loss'], logs[-1]['test_accuracy1'], logs[-1]['test_accuracy5'] = evaluate(net, test_loader)
            print("Timestamp: {timestamp} | "
                  "Iteration: {iteration:6} | "
                  "Loss: {training_loss:6.4f} | "
                  "Accuracy1: {training_accuracy1:6.4f} | "
                  "Accuracy5: {training_accuracy5:6.4f} | "
                  "Total_param: {total_param:6} | "
                  "Sparse_param: {sparse_param:6} | "
                  "Mini_Batch_Time: {mini_batch_time:6.4f} | "
                  "Test Loss: {test_loss:6.4f} | "
                  "Test Accuracy1: {test_accuracy1:6.4f} | " 
                  "Test_Accuracy5: {test_accuracy5:6.4f}".format(**logs[-1]))


    df = pd.DataFrame(logs)
    df.to_csv('./log/{}_Node{}_{}.csv'.format(args.file_name,args.dist_rank,datetime.now().strftime("%Y-%m-%d %H:%M:%S")), index_label='index')
    print ("Finished Training")


def evaluate(net, test_loader):
    top1 = AverageMeter()
    top5 = AverageMeter()
    test_loss = 0
    criterion = nn.CrossEntropyLoss()
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            output = net.forward(inputs)
            test_loss += criterion(output, labels).data.item()

            prec1, prec5 = accuracy(output.data, labels, topk=(1, 5))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

    return test_loss, top1.avg.item()/100.0, top5.avg.item()/100.0

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



if __name__ == "__main__":
    args = parser.parse_args()
    #dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.dist_rank,world_size=args.world_size)
    dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=args.world_size,group_name='mygroup')
    run()

