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
import gc

"""
 pipeline ResNet script for Tianhe-2  with gpu cluster

"""

def train(queue, layer, e, args, logger, loader=None):


    batch_size = args.batch_size
    package_size = args.package_size
    send_num = args.send_num
    epoch = 0
    #max_epoch = args.epoch
    max_epoch = 0
    all_loss = 0
    batch_idx = 0
    total = 0
    correct = 0

    access_stop_flag = False

    queue_wait = 5

    point = 0
    time_sleep = 6

    save_point = 0

    if loader is not None and (dist.get_rank() == 0 or dist.get_rank() == 6):
        data_iter = iter(loader)

    optimizer = optim.SGD(layer.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()


    try:
        while True:

            if dist.get_rank() == 0:
                package = torch.zeros([package_size, batch_size, 64, 32, 32], requires_grad=True)
                try:
                    input_v_pack = torch.zeros([package_size, batch_size, 3, 32, 32], requires_grad=True)
                    for count in range(package_size):
                        input_v, target_v = next(data_iter)
                        input_v_pack[count] = input_v
                        output_v = layer(input_v)
                        package[count] = output_v
                    input_v_pack.share_memory_()
                    queue.put(input_v_pack)
                except StopIteration as stop_e:
                    if epoch < max_epoch:
                        logger.error('rank-%s: epoch-%s start...', str(dist.get_rank()), str(epoch))
                        epoch += 1
                        data_iter = iter(loader)
                        continue
                    else:
                        logger.error('iteration end successfully......')
                        send_opt = dist.isend(tensor=torch.zeros(1), dst=1)
                        send_opt.wait()
                        e.wait()
                        break
                send_opt = dist.isend(tensor=package, dst=1)
                send_opt.wait()
                logger.error('rank 0 send.....')

            elif dist.get_rank() == 1:
                try:
                    rec_val = torch.zeros([package_size, batch_size, 64, 32, 32], requires_grad=True)
                    dist.recv(tensor=rec_val, src=0)
                except RuntimeError as error:
                    send_opt = dist.isend(tensor=torch.zeros(1), dst=2)
                    send_opt.wait()
                    e.wait()
                    break
                package = torch.zeros([package_size, batch_size, 64, 32, 32])
                for count in range(package_size):
                    package[count] = layer(rec_val[count])
                rec_val.share_memory_()
                queue.put(rec_val)
                send_opt = dist.isend(tensor=package, dst=2)
                send_opt.wait()
                del package
                gc.collect()
                logger.error('rank 1 send....')

            elif dist.get_rank() == 2:
                try:
                    rec_val = torch.zeros([package_size, batch_size, 64, 32, 32], requires_grad=True)
                    dist.recv(tensor=rec_val, src=1)
                except RuntimeError as error:
                    send_opt = dist.isend(tensor=torch.zeros(1), dst=3)
                    send_opt.wait()
                    e.wait()
                    break

                package = torch.zeros([package_size, batch_size, 128, 16, 16], requires_grad=True)
                for count in range(package_size):
                    package[count] = layer(rec_val[count])
                rec_val.share_memory_()
                queue.put(rec_val)
                send_opt = dist.isend(tensor=package, dst=3)
                send_opt.wait()
                del package
                gc.collect()
                #logging.error('rank 2 send.....')
            elif dist.get_rank() == 3:
                try:
                    rec_val = torch.zeros([package_size, batch_size, 128, 16, 16], requires_grad=True)
                    dist.recv(tensor=rec_val, src=2)
                except RuntimeError as error:
                    send_opt = dist.isend(tensor=torch.zeros(1), dst=4)
                    send_opt.wait()
                    e.wait()
                    break

                package = torch.zeros([package_size, batch_size, 256, 8, 8], requires_grad=True)
                for count in range(package_size):
                    package[count] = layer(rec_val[count])
                rec_val.share_memory_()
                queue.put(rec_val)
                send_opt = dist.isend(tensor=package, dst=4)
                send_opt.wait()
                del package
                gc.collect()
                logging.error('rank 3 send.......')
                #print("rank 3 send.....")
            elif dist.get_rank() == 4:
                try:
                    rec_val = torch.zeros([package_size, batch_size, 256, 8, 8], requires_grad=True)
                    dist.recv(tensor=rec_val, src=3)
                except RuntimeError as error:
                    send_opt = dist.isend(tensor=torch.zeros(1), dst=5)
                    send_opt.wait()
                    e.wait()
                    break

                package = torch.zeros([package_size, batch_size, 512, 4, 4], requires_grad=True)
                for count in range(package_size):
                    package[count] = layer(rec_val[count])
                rec_val.share_memory_()
                queue.put(rec_val)
                send_opt = dist.isend(tensor=package, dst=5)
                send_opt.wait()
                del package
                gc.collect()
                logger.error('rank 4 send.....')
                #print("rank 4 send.......")
            elif dist.get_rank() == 5:
                try:
                    rec_val = torch.zeros([package_size, batch_size, 512, 4, 4], requires_grad=True)
                    dist.recv(tensor=rec_val, src=4)
                except RuntimeError as error:
                    send_opt = dist.isend(tensor=torch.zeros(1), dst=6)
                    send_opt.wait()
                    e.wait()
                    break
                rec_val.share_memory_()
                queue.put(rec_val)
                send_opt = dist.isend(tensor=torch.randn(2), dst=6)
                send_opt.wait()
                logger.error('rank 5 send......')
                #print("rank 5 send....")
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
                        target_flag = True
                        while target_flag:
                            target_v_pack = torch.zeros([package_size, batch_size], dtype=torch.long)
                            try:
                                for count in range(package_size):
                                    _, target_temp = next(data_iter)
                                    target_v_pack[count] = target_temp
                                target_flag = False
                            except StopIteration as stop_e:
                                data_iter = iter(loader)
                                logger.error("rank 6 data_iter...")
                                continue

                    except Empty as empty:
                        logger.error('rank 6 prepare to end....')
                        #print("rank 6 prepare to end....")
                        send_opt = dist.isend(tensor=torch.zeros(1), dst=7)
                        send_opt.wait()
                        e.wait()
                        break
                    package = torch.zeros([package_size, batch_size, 512, 4, 4], requires_grad=True)

                    for count in range(package_size):
                        input_v = rec_pack[count].clone()
                        input_v.requires_grad_()
                        output_v = layer(input_v)
                        optimizer.zero_grad()
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
                    send_opt = dist.isend(tensor=package, dst=7)
                    send_opt.wait()
                    del input_v
                    del package
                    gc.collect()
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
                        send_opt = dist.isend(tensor=torch.zeros(1), dst=8)
                        send_opt.wait()
                        e.wait()
                        break
                    package = torch.zeros([package_size, batch_size, 256, 8, 8], requires_grad=True)
                    for count in range(package_size):
                        input_v = input_v_pack[count].clone()
                        input_v.requires_grad_()
                        back_grad = back_grad_pack[count]# requires_grad == True????????
                        output_v = layer(input_v)
                        optimizer.zero_grad()
                        output_v.backward(back_grad)
                        optimizer.step()
                        package[count] = input_v.grad
                    send_opt = dist.isend(tensor=package, dst=8)
                    send_opt.wait()
                    del input_v
                    del package
                    gc.collect()
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
                        send_opt = dist.isend(tensor=torch.zeros(1), dst=9)
                        send_opt.wait()
                        e.wait()
                        break
                    package = torch.zeros([package_size, batch_size, 128, 16, 16], requires_grad=True)
                    for count in range(package_size):
                        input_v = input_v_pack[count].clone()
                        input_v.requires_grad_()
                        back_grad = back_grad_pack[count]
                        output_v = layer(input_v)
                        optimizer.zero_grad()
                        output_v.backward(back_grad)
                        optimizer.step()
                        package[count] = input_v.grad
                    #dist.isend(tensor=package, dst=9)
                    send_opt = dist.isend(tensor=package, dst=9)
                    send_opt.wait()
                    del input_v
                    del package
                    gc.collect()
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
                        send_opt = dist.isend(tensor=torch.zeros(1), dst=10)
                        send_opt.wait()
                        e.wait()
                        break
                    package = torch.zeros([package_size, batch_size, 64, 32, 32], requires_grad=True)
                    for count in range(package_size):
                        input_v = input_v_pack[count].clone()
                        input_v.requires_grad_()
                        back_grad = back_grad_pack[count]
                        output_v = layer(input_v)
                        optimizer.zero_grad()
                        output_v.backward(back_grad)
                        optimizer.step()
                        package[count] = input_v.grad
                    send_opt = dist.isend(tensor=package, dst=10)
                    send_opt.wait()
                    del input_v
                    del package
                    gc.collect()
                    #dist.isend(tensor=package, dst=10)
                    #logger.error('rank 9 send....')
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
                        send_opt = dist.isend(tensor=torch.zeros(1), dst=11)
                        send_opt.wait()
                        e.wait()
                        break
                    package = torch.zeros([package_size, batch_size, 64, 32, 32], requires_grad=True)
                    for count in range(package_size):
                        input_v = input_v_pack[count].clone()
                        input_v.requires_grad_()
                        back_grad = back_grad_pack[count]
                        output_v = layer(input_v)
                        optimizer.zero_grad()
                        output_v.backward(back_grad)
                        optimizer.step()
                        package[count] = input_v.grad
                    send_opt = dist.isend(tensor=package, dst=11)
                    send_opt.wait()
                    del input_v
                    del package
                    gc.collect()
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
                        input_v = input_v_pack[count].clone()
                        input_v.requires_grad_()
                        back_grad = back_grad_pack[count]
                        output_v = layer(input_v)
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

    except Exception as e:
        logger.error('rank-' + str(dist.get_rank()) + ' fail: ', exc_info=True)
        #print(e)
        return






def test(layer, e, acc, args, logger, loader=None):

    batch_size = args.batch_size
    all_loss = 0
    total = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    logger.error("start eval..............")
    try:
        while True:

            if dist.get_rank() == 0:
                for id, (input_v, target_v) in enumerate(loader):
                    output_v = layer(input_v)
                    send_opt = dist.isend(tensor=output_v, dst=1)
                    send_opt.wait()
                    logger.error("eval send ........")
                logger.error('eval iteration end successfully......')
                send_opt = dist.isend(tensor=torch.zeros(1), dst=1)
                send_opt.wait()
                e.wait()
                break
            elif dist.get_rank() == 1:
                try:
                    rec_val = torch.zeros([batch_size, 64, 32, 32], requires_grad=True)
                    dist.recv(tensor=rec_val, src=0)
                except RuntimeError as error:
                    send_opt = dist.isend(tensor=torch.zeros(1), dst=2)
                    send_opt.wait()
                    e.wait()
                    break
                output_v = layer(rec_val)
                send_opt = dist.isend(tensor=output_v, dst=2)
                send_opt.wait()
                logger.error('eval: rank 1 send....')

            elif dist.get_rank() == 2:
                try:
                    rec_val = torch.zeros([batch_size, 64, 32, 32], requires_grad=True)
                    dist.recv(tensor=rec_val, src=1)
                except RuntimeError as error:
                    send_opt = dist.isend(tensor=torch.zeros(1), dst=3)
                    send_opt.wait()
                    e.wait()
                    break
                output_v = layer(rec_val)
                send_opt = dist.isend(tensor=output_v, dst=3)
                send_opt.wait()
                logging.error('eval: rank 2 send.....')
            elif dist.get_rank() == 3:
                try:
                    rec_val = torch.zeros([batch_size, 128, 16, 16], requires_grad=True)
                    dist.recv(tensor=rec_val, src=2)
                except RuntimeError as error:
                    send_opt = dist.isend(tensor=torch.zeros(1), dst=4)
                    send_opt.wait()
                    e.wait()
                    break
                output_v = layer(rec_val)
                send_opt = dist.isend(tensor=output_v, dst=4)
                send_opt.wait()
                logging.error('eval: rank 3 send.......')
            elif dist.get_rank() == 4:
                try:
                    rec_val = torch.zeros([batch_size, 256, 8, 8], requires_grad=True)
                    dist.recv(tensor=rec_val, src=3)
                except RuntimeError as error:
                    send_opt = dist.isend(tensor=torch.zeros(1), dst=5)
                    send_opt.wait()
                    e.wait()
                    break
                output_v = layer(rec_val)
                send_opt = dist.isend(tensor=output_v, dst=5)
                send_opt.wait()
                logger.error('eval: rank 4 send.....')
            elif dist.get_rank() == 5:
                try:
                    rec_val = torch.zeros([batch_size, 512, 4, 4], requires_grad=True)
                    dist.recv(tensor=rec_val, src=4)
                    for batch_idx, (_, target_v) in enumerate(loader):
                        output_v = layer(rec_val)
                        loss = criterion(output_v, target_v)
                        all_loss += loss.item()
                        _, predicted = output_v.max(1)
                        total += target_v.size(0)
                        correct += predicted.eq(target_v).sum().item()
                        logger.error('eval: batch_idx: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                            batch_idx, all_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                        rec_val = torch.zeros([batch_size, 512, 4, 4], requires_grad=True)
                        dist.recv(tensor=rec_val, src=4)
                except RuntimeError as error:
                    logger.error("rank 5 triger to stop.....")
                    if total != 0:
                        value = 100. * correct / total
                        acc.set_global_acc(value)
                    e.set()
                    break
            elif dist.get_rank() == 6:
                e.wait()
                break
            elif dist.get_rank() == 7:
                e.wait()
                break
            elif dist.get_rank() == 8:
                e.wait()
                break
            elif dist.get_rank() == 9:
                e.wait()
                break
            elif dist.get_rank() == 10:
                e.wait()
                break
            elif dist.get_rank() == 11:
                e.wait()
                break


        logger.info(' eval rank-%s stop....', str(dist.get_rank()))
    except Exception as e:
        logger.error('eval rank-' + str(dist.get_rank()) + ' fail: ', exc_info=True)
        return






def run(queue, layer, global_event, epoch_event, acc, args, train_loader=None, test_loader=None):

    logger = logging.getLogger('rank-' + str(dist.get_rank()))
    file_handler = logging.FileHandler('/WORK/sysu_wgwu_4/xpipe/XPipe/rank-' + str(dist.get_rank()) + '.log')
    file_handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter(
        fmt='%(levelname)s:%(asctime)s | pricess_id-%(process)d | %(funcName)s->%(lineno)d | %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    """
    start_epoch = 0
    epoch_num = args.epoch
    r = dist.get_rank()
    if r in [0, 1, 2, 3, 4, 5]:
        if True and os.path.isdir('checkpoint'):
            checkpoint = torch.load('./checkpoint/rank-' + str(r) + '_ckpt.t7')
            layer.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            best_acc.value = checkpoint['acc']
            logger.error("load the model: start_epoch: " + str(start_epoch) + " best_acc: " + str(best_acc.value))

    for epoch in range(start_epoch, start_epoch + epoch_num):
        logger.error("training epoch-" + str(epoch) + '.........................................................................')
        layer.train()
        train(queue, layer, epoch_event, args, logger, train_loader)
        gc.collect()
        time.sleep(1)
        epoch_event.clear()
        logger.error("eval epoch-" + str(epoch) + '.........................................................................')
        layer.eval()
        with torch.no_grad():
            test(layer, epoch_event, acc, args, logger, test_loader)
        if acc.get_global_acc() > acc.get_best_acc():
            logger.error("epoch-" + str(epoch) + ": save.........")
            state = {
                'net': layer.state_dict(),
                'acc': acc.get_global_acc(),
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if r in [0, 1, 2, 3, 4, 5]:
                torch.save(state, './checkpoint/rank-' + str(r) + '_ckpt.t7')
            if r == 5:
                acc.set_best_acc(acc.get_global_acc())
        time.sleep(1)
        epoch_event.clear()
    """

    epoch = 0
    r = dist.get_rank()
    acc.set_global_acc(2.0)
    if acc.get_global_acc() > acc.get_best_acc():
        logger.error("epoch-" + str(epoch) + ": save.........")
        state = {
            'net': layer.state_dict(),
            'acc': acc.get_global_acc(),
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if r in [0, 1, 2, 3, 4, 5]:
            torch.save(state, './checkpoint/rank-' + str(r) + '_ckpt.t7')

        acc.set_best_acc(acc.get_global_acc())
    logger.error("rank-" + str(r) + ": run method end......")
    global_event.set()



def init_processes(fn, args, queue, layer, rank, global_event, epoch_event, acc):
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

        #pin memory
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


    if rank == 5:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.data_worker, drop_last=True)

    if rank == 0:
        fn(queue, layer, global_event, epoch_event, acc, args, train_loader=trainloader, test_loader=testloader)
    elif rank == 5:
        fn(queue, layer, global_event, epoch_event, acc, args, test_loader=testloader)
    elif rank == 6:
        fn(queue, layer, global_event, epoch_event,acc, args, train_loader=trainloader, test_loader=testloader)
    else:
        fn(queue, layer, global_event, epoch_event, acc, args)


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

    #torch.manual_seed(1)

    bm.register('get_epoch_event')
    bm.register('get_global_event')
    bm.register('get_acc')

    m = bm(address=(args.ip, 5000), authkey=b'xpipe')
    m.connect()
    global_event = m.get_global_event()
    epoch_event = m.get_epoch_event()
    acc = m.get_acc()

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

    f_p = Process(target=init_processes, args=(run, args, queue, layer, args.rank, global_event, epoch_event, acc))
    f_p.start()
    b_p = Process(target=init_processes, args=(run, args, queue, layer, (11 - args.rank), global_event, epoch_event, acc))
    b_p.start()
    f_p.join()
    b_p.join()