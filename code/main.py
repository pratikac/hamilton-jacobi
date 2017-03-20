from __future__ import print_function
import argparse, math, random
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from timeit import default_timer as timer
import torch.backends.cudnn as cudnn

from exptutils import *
import models, loader, optim
import numpy as np

opt = add_args([
['-o', '/local2/pratikac/results', 'output'],
['-m', 'lenet', 'lenet | mnistfc | allcnn | wideresnet'],
['--optim', 'EntropySGD', 'EntropySGD | HJB | PME | FB | LL'],
['--dataset', 'mnist', 'mnist | rotmnist | cifar10 | cifar100'],
['--retrain', '', 'checkpoint'],
['-b', 128, 'batch_size'],
['--augment', False, 'data augmentation'],
['-B', 100, 'Max epochs'],
['--lr', 0.1, 'learning rate'],
['--lr_schedule', '', 'learning rate schedule'],
['--l2', 0.0, 'ell-2'],
['-L', 0, 'sgld iterations'],
['--eps', 1e-4, 'sgld noise'],
['--g0', 0.03, 'gamma'],
['--g1', 0.0, 'scoping'],
['-s', 42, 'seed'],
['-g', 0, 'gpu idx'],
['-l', False, 'log'],
['-v', False, 'verbose']
])
if opt['L'] > 0:
    opt['freq'] = 1
else:
    opt['freq'] = 10

th.set_num_threads(2)
th.cuda.set_device(opt['g'])
random.seed(opt['s'])
np.random.seed(opt['s'])
th.manual_seed(opt['s'])
th.cuda.manual_seed(opt['s'])
cudnn.benchmark = True

train_loader, val_loader, test_loader = getattr(loader, opt['dataset'])(opt)
model = getattr(models, opt['m'])(opt).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = getattr(optim, opt['optim'])(model.parameters(),
        config = dict(lr=opt['lr'], momentum=0.9, nesterov=True, weight_decay=opt['l2'],
        L=opt['L'], eps=opt['eps'], g0=opt['g0'], g1=opt['g1'], verbose=opt['v']))

if not opt['retrain'] == '':
    ckpt = th.load(opt['retrain'])
    model.load_state_dict(ckpt['state_dict'])
    print('Retraining model: %s'%ckpt['name'])

def schedule(e):
    if opt['lr_schedule'] == '':
        opt['lr_schedule'] = json.dumps([[opt['B'], opt['lr']]])

    lrs = json.loads(opt['lr_schedule'])

    idx = len(lrs)-1
    for i in xrange(len(lrs)):
        if e < lrs[i][0]:
            idx = i
            break
    lr = lrs[idx][1]

    print('[LR]: ', lr)
    logging.info('[LR] %.5f'%lr)
    optimizer.config['lr'] = lr

def train(e):
    schedule(e)

    model.train()

    fs, top1 = AverageMeter(), AverageMeter()
    ts = timer()

    bsz = opt['b']
    maxb = len(train_loader)

    for bi in xrange(maxb):
        def helper():
            def feval():
                x,y = next(train_loader)
                x, y = Variable(x.cuda()), Variable(y.squeeze().cuda())
                bsz = x.size(0)

                optimizer.zero_grad()
                yh = model(x)
                f = criterion.forward(yh, y)
                f.backward()

                prec1, = accuracy(yh.data, y.data, topk=(1,))
                err = 100.-prec1[0]
                return (f.data[0], err)
            return feval

        f, err = optimizer.step(helper(), model, criterion)
        th.cuda.synchronize()

        fs.update(f, bsz)
        top1.update(err, bsz)

        s = dict(i=bi + e*maxb, e=e, f=f, top1=err)
        logging.info(json.dumps(s))

        if bi % 100 == 0 and bi != 0:
            print((color('blue', '[%2d][%4d/%4d] %2.4f %2.2f%%'))%(e,bi,maxb,
                fs.avg, top1.avg))

    s = dict(e=e, i=0, f=fs.avg, top1=top1.avg)
    logging.info(json.dumps(s))

    print(  (color('blue', '++[%2d] %2.4f %2.2f%% [%.2fs]'))% (e,
            fs.avg, top1.avg, timer()-ts))

def set_dropout(cache = None, p=0):
    if cache is None:
        cache = []
        for l in model.modules():
            if 'Dropout' in str(type(l)):
                cache.append(l.p)
                l.p = p
        return cache
    else:
        for l in model.modules():
            if 'Dropout' in str(type(l)):
                assert len(cache) > 0, 'cache is empty'
                l.p = cache.pop(0)

def dry_feed():
    cache = set_dropout()
    maxb = len(train_loader)
    for bi in xrange(maxb):
        x,y = next(train_loader)
        x,y =   Variable(x.cuda(), volatile=True), \
                Variable(y.squeeze().cuda(), volatile=True)
        yh = model(x)
    set_dropout(cache)

def val(e, data_loader):
    dry_feed()
    model.eval()

    maxb = len(data_loader)
    fs, top1 = AverageMeter(), AverageMeter()
    for bi in xrange(maxb):
        x,y = next(data_loader)
        bsz = x.size(0)

        x,y =   Variable(x.cuda(), volatile=True), \
                Variable(y.squeeze().cuda(), volatile=True)
        yh = model(x)

        f = criterion.forward(yh, y).data[0]
        prec1, = accuracy(yh.data, y.data, topk=(1,))
        err = 100-prec1[0]

        fs.update(f, bsz)
        top1.update(err, bsz)

    s = dict(e=e, i=0, f=fs.avg, top1=top1.avg)
    logging.info(json.dumps(s))
    print((color('red', '**[%2d] %2.4f %2.4f%%\n'))%(e, fs.avg, top1.avg))
    print('')


def main():
    print(opt)
    build_filename(opt, blacklist=['lr_schedule','retrain','step', \
                                'ratio','freq','v','dataset', 'augment', 'd'])
    create_logger(opt)
    for e in xrange(opt['B']):
        train(e)
        if e % opt['freq'] == opt['freq'] -1:
            val(e, val_loader)
        #save(model, opt)

    # print(color('red', 'Test error: '))
    # val(e, test_loader)

main()