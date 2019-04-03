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