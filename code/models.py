import torch as th
import torch.nn as nn
import math, logging, pdb


class View(nn.Module):
    def __init__(self,o):
        super(View, self).__init__()
        self.o = o
    def forward(self,x):
        return x.view(-1, self.o)

def num_parameters(model):
    return sum([w.numel() for w in model.parameters()])

class mnistfc(nn.Module):
    def __init__(self, opt):
        super(mnistfc, self).__init__()
        self.name = 'mnsitfc'

        c = 1024
        opt['d'] = 0.5
        opt['l2'] = 0.

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
        opt['l2'] = 0.

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
        opt['l2'] = 1e-3

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

class caddtable_t(nn.Module):
    def __init__(self, m1, m2):
        super(caddtable_t, self).__init__()
        self.m1, self.m2 = m1, m2

    def forward(self, x):
        return self.m1(x) + self.m2(x)

class wideresnet(nn.Module):
    def __init__(self, opt = {'d':0., 'depth':16, 'widen':2}):
        super(wideresnet, self).__init__()
        self.name = 'wideresnet'

        depth = opt.get('depth', 16)
        widen = opt.get('widen', 2)
        opt['l2'] = 5e-4

        nc = [16, 16*widen, 32*widen, 64*widen]
        assert (depth-4)%6 == 0, 'Incorrect depth'
        n = (depth-4)/6

        if opt['dataset'] == 'cifar10':
            num_classes = 10
        elif opt['dataset'] == 'cifar100':
            num_classes = 100

        def block(ci, co, s, p=0.):
            h = nn.Sequential(
                    nn.Sequential(nn.BatchNorm2d(ci),
                    nn.ReLU(inplace=True)),
                    nn.Conv2d(ci, co, kernel_size=3, stride=s, padding=1),
                    nn.BatchNorm2d(co),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(co, co, kernel_size=3, stride=1, padding=1))
            if ci == co:
                return caddtable_t(h, nn.Sequential())
            else:
                return caddtable_t(h,
                        nn.Conv2d(ci, co, kernel_size=1, stride=s))

        def netblock(nl, ci, co, blk, s, p=0.):
            ls = [blk(i==0 and ci or co, co, i==0 and s or 1, p) for i in xrange(nl)]
            return nn.Sequential(*ls)

        self.m = nn.Sequential(
                nn.Conv2d(3, nc[0], kernel_size=3, stride=1, padding=1),
                netblock(n, nc[0], nc[1], block, 1),
                netblock(n, nc[1], nc[2], block, 2),
                netblock(n, nc[2], nc[3], block, 2),
                nn.BatchNorm2d(nc[3]),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(8),
                View(nc[3]),
                nn.Linear(nc[3], num_classes))

        # initialize weights
        for m in self.m.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)