import torch
import matplotlib.pyplot as plt
import numpy as np
from visdom import Visdom
import os
from collections import Counter
lines = []
with open('../new_grad.txt', 'r+') as f:
    line = f.readline()
l = [float(x.strip()) for x in line[7: len(line) -3].split(',')]
tensor = torch.Tensor(l)
vis = Visdom()
#vis.bar(X=tensor.view(-1).numpy())
vis.line(Y=tensor.view(-1).numpy()[0:])
print(tensor.view(-1).numpy()[0:100])
#vis.line(Y=np.array(list(range(50))))
