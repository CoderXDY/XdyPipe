def quantize(input, num_bits=8, half=True, residual=None):
    num_chunks = input.shape[0]
    B = input.shape[0]
    y = input.view(B // num_chunks, -1)
    min_value = y.min(-1)[0].mean(-1)
    max_value = y.max(-1)[0].mean(-1)
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    scale = (max_value - min_value) / (qmax - qmin)
    scale = max(scale, 1e-8)
    input.add_(-min_value).div_(scale).add_(qmin)
    input.clamp_(qmin, qmax).round_()
    input = input.view(-1)

    # max_sign = torch.ones([1]) if max_value > 0 else torch.zeros([1])
    # min_sign = torch.ones([1]) if min_value > 0 else torch.zeros([1])
    # tensor = torch.cat([input, (max_value.abs_()*10000).view(1), max_sign.cuda(),
    #                     (min_value.abs_()*10000).view(1), min_sign.cuda()])
    # return tensor.byte()
    tensor = torch.cat([input, scale.view(1), min_value.view(1)])
    if half:
        return tensor.half()
    else:
        return tensor

def dequantize(input, shape, num_bits=8):
    if input.type() != 'torch.FloatTensor':
        input = input.float()
    min_value = input[-1]
    scale = input[-2]
    input = input[0: -2].view(shape)
    qmin = 0.
    input.add_(-qmin).mul_(scale).add_(min_value)
    return input
    # qmin = 0.
    # qmax = 2. ** num_bits - 1.
    # input = input.float()
    # max_value, max_sign, min_value, min_sign = input[-4], input[-3], input[-2], input[-1]
    # max_value = max_value / 10000 * 1 if max_sign == 1 else max_value / 10000 * (-1)
    # min_value = min_value / 10000 * 1 if min_sign == 1 else min_value / 10000 * (-1)
    # input = input[0: -4].view(shape)
    # scale = (max_value - min_value) / (qmax - qmin)
    # scale = max(scale, 1e-8)
    # input.add_(-qmin).mul_(scale).add_(min_value)
    # return input










input = torch.mean(input, dim=0, keepdim=True)




"""
def quantize(input, num_bits=8, byte=True, residual=None):
    num_chunks = input.shape[0]
    B = input.shape[0]
    y = input.view(B // num_chunks, -1)
    min_value = y.min(-1)[0].mean(-1)
    max_value = y.max(-1)[0].mean(-1)
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    scale = (max_value - min_value) / (qmax - qmin)
    scale = max(scale, 1e-8)
    input.add_(-min_value).div_(scale).add_(qmin)
    input.clamp_(qmin, qmax).round_()
    input = input.view(-1).cpu()
    scale = scale.cpu()
    min_value = min_value.cpu()
    if byte:
        input = input.byte()
    return input.numpy(), scale.numpy(), min_value.numpy()

def dequantize(tuple, shape, num_bits=8):
    tensor, scale, min_value = torch.from_numpy(tuple[0]).view(shape), \
                               torch.from_numpy(tuple[1]), torch.from_numpy(tuple[2])
    qmin = 0.
    tensor = tensor.float() if tensor.type() == 'torch.ByteTensor' else tensor
    tensor.add_(-qmin).mul_(scale).add_(min_value)
    return tensor

"""



def my_quantize(input, nor_bit=16, num_bits=8, half=True, residual=None):

    #first quantize input to nor_bit float-point
    num_chunks = input.shape[0]
    B = input.shape[0]
    y = input.view(B // num_chunks, -1)
    min_value = y.min(-1)[0].mean(-1)
    max_value = y.max(-1)[0].mean(-1)
    qmin = 0.
    qmax = 2. ** nor_bit - 1.
    input.mul_(qmax -qmin).div_(max_value-min_value)
    #sencond to quantize....
    num_chunks2 = input.shape[0]
    B2 = input.shape[0]
    y2 = input.view(B2 // num_chunks2, -1)
    min_value2 = y2.min(-1)[0].mean(-1)
    max_value2 = y2.max(-1)[0].mean(-1)
    threshold = (max_value2 - min_value2) / 3
    #if < threshold clip
    input[input < threshold] = 0.
    #else:
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    scale = (max_value2 - min_value2) / (qmax - qmin)
    input.add_(-min_value2).div_(scale).add_(qmin)
    input.clamp_(qmin, qmax).round_()
    input = input.view(-1)
    #third: to sparse
    indexs = input.nonzero().t()
    values = input[indexs[0]]
    sparse_tensor = torch.cat([indexs[0].float(), values,
                               scale.view(1), min_value2.view(1),
                               max_value.view(1), min_value.view(1)])

    if half:
        return sparse_tensor.half()
    else:
        return sparse_tensor




def my_dequantize(input, shape, num_bits=8, nor_bit=16):
    if input.type() != 'torch.FloatTensor':
        input = input.float()
    min_value = input[-1]
    max_value = input[-2]
    min_value2 = input[-3]
    scale = input[-4]

    half_size = int(len(input[0:-4]) / 2)
    indexs = input[: half_size].view(1, half_size).long()
    values = input[half_size: -4]
    length = 1
    for i in range(len(shape)):
        length *= shape[i]
    sparse_tensor = torch.sparse.FloatTensor(indexs, values, torch.Size([length]))
    input = sparse_tensor.to_dense().view(shape)


    qmin = 0.
    input.add_(-qmin).mul_(scale).add_(min_value2)
    input[input == 0.] = min_value2
    input.mul_(max_value -min_value).div_(2. ** nor_bit - 1.)
    return input




"""

my quantized scheme 

"""
def quantize(input, num_bits=8, half=True, residual=None):
    sign = input.sign()
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.
    scale = qmax - qmin
    input_abs = torch.abs(input)
    max_val = torch.max(input_abs)
    input = torch.round(input_abs.mul(scale).div(max_val)).mul_(sign)
    input = input.view(-1)
    tensor = torch.cat([input, max_val.view(1)])
    if half:
        return tensor.half()
    else:
        return tensor

    #b = torch.abs(a)
    #c = torch.max(b)
    #torch.round(torch.abs(a).mul(255).div(c))

def dequantize(input, shape, num_bits=8):
    if input.type() != 'torch.FloatTensor':
        input = input.float()
    max_val = input[-1]
    input = input[0: -1].view(shape)
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.
    scale = qmax - qmin
    input.mul_(max_val).div_(scale)
    return input


"""
根据上面的模式增加均值共同属性来生成max_val用于反量化
"""
def quantize(input, num_bits=8, half=True, residual=None):
    sign = input.sign()
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.
    sum = max(input.sum(), qmax)
    scale = qmax - qmin
    input_abs = torch.abs(input)
    max_val = torch.max(input_abs)
    input = torch.round(input_abs.mul(scale).div(max_val)).mul_(sign)
    input = input.view(-1)
    tensor = torch.cat([input, torch.tensor(sum).cuda().view(1)])
    return tensor

    #b = torch.abs(a)
    #c = torch.max(b)
    #torch.round(torch.abs(a).mul(255).div(c))

def dequantize(input, shape, num_bits=8):
    if input.type() != 'torch.FloatTensor':
        input = input.float()
    sum = input[-1]
    input = input[0: -1].view(shape)
    input_sum = input.sum()
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.
    scale = qmax - qmin
    prop = sum.div(input_sum)
    max_val = qmax / prop
    input.mul_(max_val).div_(scale)
    return input


a = torch.randn([3,2,2])
s = a.sign()
a_abs = a.abs()
a_mean = a_abs.mean()
scale = a_mean.div(2**6)
x = torch.round((a_abs - a_mean).mul(2**6).div(a_mean).add(2**6))

print("prop: " + str(x.var() / a.var()))
#a_copy = a_abs.clone()
#a_copy[a_copy >= a_mean] = torch.round((a_copy - a_mean).mul(2**6).div(a_mean).add(2**6))
#a_max = torch.max(a_abs)
#z = torch.round(a_abs.mul(127).div(a_max))




a = torch.randn([3,2,2])
s = a.sign()
a_max = torch.max(torch.abs(a).div(a.sum()))
x = torch.round(a.mul(255).div(a_max)).mul(s)

x_max = torch.max(torch.abs(x).div(x.sum()))

y = x.mul(x_max).div(255)



### final version beta 01 no error feeback ###
def quantize(input, num_bits=8, half=True, residual=None):
    sign = input.sign()
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.
    scale = qmax - qmin
    input_abs = torch.abs(input)
    max_val = torch.max(input_abs) if (torch.max(input_abs) * 10000) < qmax else torch.tensor(qmax / 10000).cuda(1)
    input = torch.round(input_abs.mul(scale).div(max_val)).mul_(sign)
    input = input.view(-1)
    tensor = torch.cat([input, max_val.mul(10000).view(1)])
    return tensor

    #b = torch.abs(a)
    #c = torch.max(b)
    #torch.round(torch.abs(a).mul(255).div(c))

def dequantize(input, shape, num_bits=8):
    if input.type() != 'torch.FloatTensor':
        input = input.float()
    max_val = input[-1].div(10000)
    input = input[0: -1].view(shape)
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.
    scale = qmax - qmin
    input.mul_(max_val).div_(scale)
    return input



### final version beata has error feeback ,but ....
def quantize(input, num_bits=8, residual=None):
    residual = torch.zeros(input.size()).cuda(1) if residual is None else residual
    sign = input.sign()
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.
    scale = qmax - qmin
    input.add_(residual)
    input_abs = torch.abs(input)
    max_val = torch.max(input_abs) if (torch.max(input_abs) * 10000) < qmax else torch.tensor(qmax / 10000).cuda(1)
    input = torch.round(input_abs.mul(scale).div(max_val)).mul_(sign)
    #calculate the residual
    de_input = input.mul(max_val).div(scale)
    residual = input - de_input

    input = input.view(-1)
    tensor = torch.cat([input, max_val.mul(10000).view(1)])
    return tensor, residual

    #b = torch.abs(a)
    #c = torch.max(b)
    #torch.round(torch.abs(a).mul(255).div(c))

def dequantize(input, shape, num_bits=8):
    if input.type() != 'torch.FloatTensor':
        input = input.float()
    max_val = input[-1].div(10000)
    input = input[0: -1].view(shape)
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.
    scale = qmax - qmin
    input.mul_(max_val).div_(scale)
    return input



a = torch.randn([3,2,2])
s = a.sign()
b = torch.abs(a)
c = torch.max(b)
d = torch.round(b.mul(255).div(c)).mul(s)
a_d = a.div(c)

d_m = torch.max(torch.abs(d))
d_d = d.div(d_m)

print(a_d)
print("----")
print(d_d)

########################
a = torch.randn([3,2,2])
print(a)

q = torch.round(a.mul(255).mul(1000))
print("-----------")
dq = q.div(1000 * 255)
print(dq)

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





"""
    # q_part1 = torch.round(input_abs.mul(2).div(threshold).mul(sign))
    # q_part2 = torch.round(input_abs.mul(fra).add(2).mul(sign))
    # 
    # new_input = torch.where(input_abs < threshold, q_part1, q_part2)
"""








def piecewise_quantize(input, num_bits=8, residual=None, prop=1000):
    sign = input.sign()
    ab = input.abs()
    v_max = torch.max(ab)
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.
    tensor = ab.mul(qmax - qmin).div(v_max).mul(sign).cpu()

    return torch.cat([tensor.view(-1), torch.Tensor([v_max])]), None
def de_piecewise_quantize(input, shape, num_bits=8, prop=1000):
    input, v_max = input[0: -1], input[-1]
    input = input.view(shape)
    sign = input.sign()
    ab = input.abs()
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.
    return ab.mul(v_max).div(qmax - qmin).mul(sign)



def piecewise_quantize(input, num_bits=8, residual=None, prop=1000):
    qmin = 2.
    qmax = 2. ** (num_bits - 1) - 1.

    scale = qmax - qmin

    fraction_mul_qval = torch.round(input.mul(scale).mul(prop))
    ####
    # input.abs_()
    # fraction_mul_qval2 = input.mul(scale)#.mul(prop)
    # v_max = torch.max(torch.abs(input))
    # print(v_max)
    # print("-----")
    # fra = max((v_max - threshold), 1e-8)
    # print(fra)
    #
    # q2 = torch.round(fraction_mul_qval2.div(fra)) + 2
    # idx = q2.lt(2)
    # temp = fraction_mul_qval2.div(fra)[idx]
    # print(temp)
    ####
    threshold = torch.max(torch.abs(input)) / 3
    new_input = torch.where(torch.abs(input) < threshold, fraction_mul_qval, torch.zeros(input.size(), device=torch.device('cuda:0')))


    return torch.cat([new_input.view(-1).cpu(), torch.Tensor([threshold])]), None


    # if not torch.is_tensor(residual):
    #     residual = torch.zeros(input.size(), device=torch.device('cuda:0'))
    #
    # input.add_(residual)
    #
    # qmin = 2.
    # qmax = 2. ** (num_bits - 1) - 1.
    #
    # scale = qmax - qmin
    # sign = input.sign()
    # input_abs = torch.abs(input)
    # v_max = torch.max(input_abs)
    # threshold = v_max / 3
    # dis = max((v_max - threshold), 1e-8)
    # fra = scale / dis
    # print("vmax:" + str(v_max) + "dis:" + str(dis)  + " thr:" + str(threshold))
    # part1_idx = input_abs.lt(threshold)
    # input_abs[part1_idx] = torch.round(input_abs[part1_idx].mul(2).div(threshold))
    # part2_idx = input_abs.ge(threshold)
    # input_abs[part2_idx] = torch.round(input_abs[part2_idx].mul(fra)) #.add(2)
    # input_abs.mul_(sign)
    #
    #
    # residual = torch.where(input_abs == 0, input, torch.zeros(input.size(), device=torch.device('cuda:0')))
    #
    #
    #
    # print(input_abs[input_abs.eq(0)].size())
    # result = torch.cat([input_abs.view(-1).cpu(), torch.Tensor([v_max])])
    # print("result_max: " + str(result[-1]))
    # return result, residual



def de_piecewise_quantize(input, shape, num_bits=8, prop=1000):
    input, v_max = input[0: -1], input[-1]
    input = input.view(shape)
    input = input.float()
    qmin = 2.
    qmax = 2. ** (num_bits - 1) - 1.
    scale = qmax - qmin
    input.div_(prop * scale)
    # two_tensor = input.div(prop * scale)
    # random_tensor = torch.from_numpy(np.random.randint(0, 2, size=input.size())).float().cuda()
    # one_tensor = torch.where(input == 0, random_tensor.mul(threshold).div(2), torch.zeros(input.size(), device=torch.device('cuda:0')))
    #
    # new_input = torch.where(input != 0, two_tensor, one_tensor)
    #
    # return new_input
    return input


    # qmin = 2.
    # qmax = 2. ** (num_bits - 1) - 1.
    # scale = qmax - qmin
    # input, v_max = input[0: -1], input[-1]
    # print("v_max:" + str(v_max))
    # input = input.view(shape)
    # threshold = v_max / 3
    # dis = max((v_max - threshold), 1e-8)
    # sign = input.sign()
    # input_abs = input.abs()
    # fra = dis / scale
    # part1_idx = input_abs.le(2)#lt if neet to sub 2
    # input_abs[part1_idx] = input_abs[part1_idx].mul(threshold).div(2)#set 0???
    # part2_idx = input_abs.gt(2)
    # input_abs[part2_idx] = input_abs[part2_idx].mul(fra)#- 2
    # input_abs.mul_(sign)
    #
    # return input_abs












































"""

gpipe3

"""
def train(layer, logger, shapes, args, e, data_size, trainloader, grad_queue, grad_queue2):
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
                    print("batch_idx: " + str(batch_idx))
                    inputs, targets = next(data_iter)
                    inputs = inputs.cuda()
                    outputs = layer(inputs)
                    outputs_queue.put(outputs)
                    dist.send(tensor=outputs.cpu(), dst=1)
                    print("batch end...")
                except StopIteration as stop_e:
                    send_opt = dist.isend(tensor=torch.zeros(0), dst=1)
                    send_opt.wait()
                    ###
                    while not outputs_queue.empty():
                        # try:
                        # grad_recv = torch.zeros(shapes[0])
                        # dist.recv(tensor=grad_recv, src=1)

                        # except RuntimeError as error:
                        # pass
                        try:
                            grad_recv = grad_queue.get(block=True, timeout=4)
                        except Empty as empty:
                            time.sleep(2)
                            try:
                                grad_recv = grad_queue.get(block=True, timeout=6)
                            except Empty as empty:
                                print("grad queue empty!")
                                break
                        grad_recv = torch.from_numpy(grad_recv)
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
                print("backward idx:" + str(batch_idx))
                # grad_recv = torch.zeros(shapes[0])
                # dist.recv(tensor=grad_recv, src=1)
                try:
                    grad_recv = grad_queue.get(block=True, timeout=4)
                except Empty as empty:
                    time.sleep(2)
                    try:
                        grad_recv = grad_queue.get(block=True, timeout=6)
                    except Empty as empty:
                        print("grad queue empty!")
                        break

                grad_recv = torch.from_numpy(grad_recv)
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
                    print("batch_idx:" + str(batch_idx))
                    rec_val = torch.zeros(shapes[0])
                    dist.recv(tensor=rec_val, src=0)
                    rec_val = rec_val.cuda()
                    rec_val.requires_grad_()
                    outputs = layer(rec_val)
                    outputs_queue.put([rec_val, outputs])
                    send_opt = dist.isend(tensor=outputs.cpu(), dst=2)
                    send_opt.wait()
                    print("after send....")
                except RuntimeError as error:
                    print(error)
                    while not outputs_queue.empty():
                        # grad_recv = torch.zeros(shapes[1])
                        # dist.recv(tensor=grad_recv, src=2)
                        try:
                            grad_recv = grad_queue2.get(block=True, timeout=4)
                        except Empty as empty:
                            time.sleep(2)
                            try:
                                grad_recv = grad_queue2.get(block=True, timeout=6)
                            except Empty as empty:
                                print("grad queue2 empty")
                                break
                        grad_recv = torch.from_numpy(grad_recv)
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
                        grad_queue.put(inputs.grad.cpu().numpy())
                    e.wait()
                    break
                if (batch_idx + 1) % 3 == 0:
                    back_flag = True
            else:
                print("backward batch_idx:" + str(batch_idx))
                # grad_recv = torch.zeros(shapes[1])
                # dist.recv(tensor=grad_recv, src=2)
                try:
                    grad_recv = grad_queue2.get(block=True, timeout=4)
                except Empty as empty:
                    time.sleep(2)
                    try:
                        grad_recv = grad_queue2.get(block=True, timeout=6)
                    except Empty as empty:
                        print("grad queue2 empty")
                        break
                grad_recv = torch.from_numpy(grad_recv)
                grad_recv = grad_recv.cuda(0)
                print("after recv........")
                try:
                    inputs, outputs = outputs_queue.get(block=True, timeout=4)
                except Empty:
                    print("empty........")
                    break
                inputs.requires_grad_()
                optimizer.zero_grad()
                outputs.backward(grad_recv)
                optimizer.step()
                # dist.send(tensor=inputs.grad.cpu(), dst=0)
                try:
                    grad_queue.put(inputs.grad.cpu().numpy(), timeout=4)
                except Full as full:
                    time.slee(2)
                    try:
                        grad_queue.put(inputs.grad.cpu().numpy(), timeout=6)
                    except Full as full:
                        print('full')
                        break
                if (batch_idx + 1) % 3 == 0:
                    back_flag = False
            batch_idx += 1
        elif dist.get_rank() == 2:
            rec_val = torch.zeros(shapes[1])
            dist.recv(tensor=rec_val, src=1)
            index = 0
            for batch_idx, (_, targets) in enumerate(trainloader):
                rec_val = rec_val.cuda(0)
                rec_val.requires_grad_()
                outputs = layer(rec_val)
                # start to backward....
                targets = targets.cuda(0)
                if (batch_idx + 1) % 3 == 0:
                    print("backward......." + str(index))
                    loss = criterion(outputs, targets)
                    outputs_queue.put([loss, rec_val])
                    count = 0
                    print("backeard after put.....")
                    index += 1
                    while count < 3:

                        loss, rec_val = outputs_queue.get()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        '''
                        train_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                        progress_bar(batch_idx, data_size, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                        logger.error("train:" + str(train_loss / (batch_idx + 1)))
                        acc_str = "tacc: %.3f" % (100. * correct / total,)
                        logger.error(acc_str)
                        '''
                        # send_opt = dist.isend(tensor=rec_val.grad.cpu(), dst=1)
                        # send_opt.wait()
                        try:
                            grad_queue2.put(rec_val.grad.cpu().numpy(), timeout=4)
                        except Full as full:
                            time.sleep(2)
                            try:
                                grad_queue2.put(rec_val.grad.cpu().numpy(), timeout=4)
                            except Full as full:
                                print("full...")
                                break
                        print("index: " + str(index))
                        count += 1
                        index += 1
                else:
                    print("index:" + str(index))
                    loss = criterion(outputs, targets)
                    outputs_queue.put([loss, rec_val])
                    try:
                        rec_val = torch.zeros(shapes[1])
                        dist.recv(tensor=rec_val, src=1)
                        print("after index " + str(index))
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
                            try:
                                grad_queue2.put(rec_val.grad.cpu().numpy(), timeout=4)
                            except Full as full:
                                time.sleep(2)
                                try:
                                    grad_queue2.put(rec_val.grad.cpu().numpy(), timeout=4)
                                except Full as full:
                                    print("full...")
                                    break
                            # send_opt = dist.isend(tensor=rec_val.grad.cpu(), dst=1)
                            # send_opt.wait()
                        e.wait()
                        break
                    index += 1