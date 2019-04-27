def piecewise_quantize(input, num_bits=8, fra_bits=2, residual=None, prop=1000, drop=0.99):
    qmin = 2. ** fra_bits
    qmax = 2. ** num_bits - 1.

    scale = qmax - qmin

    fraction_mul_qval = torch.round(input.mul(scale).mul(prop))


    threshold = torch.max(torch.abs(input)) * drop

    id_part1 = input.abs().lt(threshold)
    id_part2 = input.abs().ge(threshold)
    a_input = input.clone()
    a_input[id_part1] = torch.round(input[id_part1].mul(qmin).mul(prop))
    a_input[id_part2] = fraction_mul_qval[id_part2]
    return a_input
    #print(input[input.eq(0.)].size())




def de_piecewise_quantize(input, num_bits=8, fra_bits= 2, prop=1000):

    qmin = 2. ** fra_bits
    qmax = 2. ** num_bits - 1.
    scale = qmax - qmin
    a_input = input.clone()
    a_input[input.abs().le(qmin)] = input[input.abs().le(qmin)].div(qmin * prop)
    a_input[input.abs().gt(qmin)] = input[input.abs().gt(qmin)].div(scale * prop)
    return a_input


##################################################
def quan(input, num_bits=8, fra_bits=1, drop=0.8):
    qmin = 2. ** fra_bits
    qmax = 2. ** num_bits - 1
    scale = qmax - qmin
    sign = input.sign()
    ab = input.abs()
    v_max = torch.max(ab)
    threshold = torch.max(ab) * drop
    dis = max((v_max - threshold), 1e-8)
    fra = scale / dis
    id_part1 = ab.lt(threshold)
    id_part2 = ab.ge(threshold)
    ab[id_part1] = torch.round(ab[id_part1].mul(qmin).div(threshold))
    ab[id_part2] = torch.round(ab[id_part2].mul(fra))
    return ab.mul(sign), v_max, threshold


def dequan(input, v_max, threshold, num_bits=8, fra_bits=1):
    qmin = 2. ** fra_bits
    qmax = 2. ** num_bits - 1
    scale = qmax - qmin
    dis = max((v_max - threshold), 1e-8)
    sign = input.sign()
    ab = input.abs()
    fra = dis / scale
    part1_idx = ab.le(qmin)
    part2_idx = ab.gt(qmin)
    ab[part1_idx] = ab[part1_idx].mul(threshold).div(qmin)
    ab[part2_idx] = ab[part2_idx].mul(fra)
    return ab.mul(sign)

#################
def quantize(input, num_bits=8, fra_bits=1, drop=0.8):
    num_chunks = input.shape[0]
    B = input.shape[0]
    y = input.view(B // num_chunks, -1)
    min_value = y.min(-1)[0].mean(-1)
    max_value = y.max(-1)[0].mean(-1)
    qmin = 2. ** fra_bits
    qmax = 2. ** num_bits - 1.
    scale = (max_value - min_value) / (qmax - qmin)
    scale = max(scale, 1e-8)
    part2_input = input.add(-min_value).div(scale).add(qmin)
    part2_input = input.clamp(qmin, qmax).round()

    threshold = torch.max(torch.abs(input)) * drop
    id_part1 = input.abs().lt(threshold)
    id_part2 = input.abs().ge(threshold)

    new_input = input.clone()
    new_input[id_part1] = 0.
    new_input[id_part2] = part2_input[id_part2]
    return part2_input, new_input, min_value, scale, threshold


def dequantize(input, min_value, scale, threshold, num_bits=8, fra_bits=1):
    qmin = 2. ** fra_bits

    part2_input = input.add(-qmin).mul(scale).add(min_value)
    id_part1 = input.abs().lt(qmin)
    id_part2 = input.abs().ge(qmin)
    new_input = input.clone()
    new_input[id_part1] = 0.
    new_input[id_part2] = part2_input[id_part2]
    return new_input

########
def qg(input, num_bits=8, fra_bits=7, drop=0.8):
    qmin = 2. ** fra_bits
    qmax = 2. ** num_bits - 1.
    scale = qmax - qmin
    sign = input.sign()
    ab = input.abs()
    v_max = torch.max(ab)
    threshold = v_max * drop

    id_part1 = ab.lt(threshold)
    id_part2 = ab.ge(threshold)
    ab[id_part1] = ab[id_part1].mul(qmin).div(threshold)
    fra = scale / (v_max - threshold)
    ab[id_part2] = qmax - ((v_max - ab[id_part2]).mul(fra))
    ab.round_()
    return ab.mul(sign), v_max, threshold


def dqg(input, v_max, threshold, num_bits=8, fra_bits=7):
    qmin = 2. ** fra_bits
    qmax = 2. ** num_bits - 1.
    scale = qmax - qmin
    sign = input.sign()
    ab = input.abs()
    id_part1 = ab.lt(qmin)
    id_part2 = ab.ge(qmin)
    ab[id_part1] = ab[id_part1].mul(threshold).div(qmin)
    fra = (v_max - threshold) / scale
    ab[id_part2] = v_max - ((qmax - ab[id_part2]).mul(fra))
    return ab.mul(sign)