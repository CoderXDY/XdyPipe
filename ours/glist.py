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