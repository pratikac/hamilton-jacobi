import torch as th
import torch.nn as nn
from torch.autograd import Variable
import math
import logging

class View(nn.Module):
    def __init__(self,o):
        super(View, self).__init__()
        self.o = o
    def forward(self,x):
        return x.view(-1, self.o)

def num_parameters(model):
    n = 0
    for m in model.modules():
        for w in m.parameters():
            n += w.numel()
    return n

class mnistfc(nn.Module):
    def __init__(self, opt):
        super(mnistfc, self).__init__()
        self.name = 'mnsitfc'

        c = 1024
        opt['d'] = 0.5

        self.m = nn.Sequential(
            View(784),
            nn.Dropout(0.2),
            nn.Linear(784,c),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(c,c),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(c,10))

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class lenet(nn.Module):
    def __init__(self, opt):
        super(lenet, self).__init__()
        self.name = 'lenet'
        opt['d'] = 0.25

        def convbn(ci,co,ksz,psz,p):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz),
                nn.ReLU(True),
                nn.MaxPool2d(psz,stride=psz),
                nn.BatchNorm2d(co),
                nn.Dropout(p))

        self.m = nn.Sequential(
            convbn(1,20,5,3,opt['d']),
            convbn(20,50,5,2,opt['d']),
            View(50*2*2),
            nn.Linear(50*2*2, 500),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(500,10))

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class rotlenet(nn.Module):
    def __init__(self, opt):
        super(rotlenet, self).__init__()
        self.name = 'rotlenet'
        opt['d'] = 0.3

        def convpool(ci,co,ksz,psz,pstr,p):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz),
                nn.ReLU(True),
                nn.MaxPool2d(psz,stride=pstr),
                nn.BatchNorm2d(co),
                nn.Dropout(p))
        def conv(ci,co,ksz,p):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz),
                nn.ReLU(True),
                nn.BatchNorm2d(co),
                nn.Dropout(p))

        self.m = nn.Sequential(
            conv(1,20,3,opt['d']),
            convpool(20,20,3,2,2,0),
            conv(20,20,3,opt['d']),
            conv(20,20,3,opt['d']),
            conv(20,20,3,opt['d']),
            conv(20,20,3,opt['d']),
            conv(20,10,4,0),
            View(10))

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)


class allcnn(nn.Module):
    def __init__(self, opt = {'d':0.5}, c1=96, c2= 192):
        super(allcnn, self).__init__()
        self.name = 'allcnn'
        opt['d'] = 0.5

        def convbn(ci,co,ksz,s=1,pz=0):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
                nn.BatchNorm2d(co),
                nn.ReLU(True))
        self.m = nn.Sequential(
            nn.Dropout(0.2),
            convbn(3,c1,3,1,1),
            convbn(c1,c1,3,1,1),
            convbn(c1,c1,3,2,1),
            nn.Dropout(opt['d']),
            convbn(c1,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,2,1),
            nn.Dropout(opt['d']),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,10,1,1),
            nn.AvgPool2d(8),
            View(10))

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)