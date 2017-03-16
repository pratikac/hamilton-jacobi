from models import *
import torch as th
import torch.nn as nn
from operator import itemgetter
from future.utils import iteritems
import numpy as np
import collections
import pdb

m = lenet({})
#m = nn.Sequential(nn.Linear(5,3,2))

def flatten_params(model):
    return th.cat([param.data.view(-1) for param in model.parameters()], 0)

def unflatten_params(model, flattened):
    offset = 0
    for param in model.parameters():
        param.data.copy_(flattened[offset:offset + param.nelement()]).view(param.size())
        offset += param.nelement()

def check_flatten():
    for i in xrange(1000):
        x = flatten_params(m)
        unflatten_params(m, x)
    return x