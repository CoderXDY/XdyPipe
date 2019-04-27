import torch
import matplotlib.pyplot as plt
import numpy as np
from visdom import Visdom
import os
from collections import Counter
lines = []
with open('new_grad.txt', 'r+') as f:
    for line in f:
        lines.append(line)
ls = []
for line in lines:
    l = [float(x.strip()) for x in line[7: len(line) -3].split(',')]
    ls.append(torch.Tensor(l))


def piecewise_quantize(input, num_bits=8, fra_bits=2, residual=None, prop=1000, drop=0.8):
    qmin = 2. ** fra_bits
    qmax = 2. ** num_bits - 1.

    scale = qmax - qmin

    fraction_mul_qval = torch.round(input.mul(scale).mul(prop))


    threshold = torch.max(torch.abs(input)) * drop

    id_part1 = input.lt(threshold)
    id_part2 = input.ge(threshold)
    input[id_part1] = torch.round(input[id_part1].mul(qmin).mul(prop))
    input[id_part2] = fraction_mul_qval[id_part2]
    return input
    #print(input[input.eq(0.)].size())




def de_piecewise_quantize(input, num_bits=8, fra_bits= 1, prop=1000):

    qmin = 2. ** fra_bits
    qmax = 2. ** num_bits - 1.
    scale = qmax - qmin
    input[input.le(qmin)] = input[input.le(qmin)].div(qmin * prop)
    input[input.gt(qmin)] = input[input.gt(qmin)].div(scale * prop)
    return input



