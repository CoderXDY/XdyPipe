import torch
from torch.autograd import Variable as V
import torch.distributed as dist
from torch import nn as nn
import argparse
from torch.multiprocessing import Process, Queue, Value
import torch.nn.functional as F
import torch.optim as optim
import logging as Log
import time


class SubLayer(nn.Module):
    def __init__(self):
        super(SubLayer, self).__init__()

