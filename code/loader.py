import torch as th
from torchvision import datasets, transforms
import torch.utils.data
import numpy as np
import os, sys, pdb

class sampler_t:
    def __init__(self, batch_size, x,y, train=True, augment=False):
        self.n = x.size(0)
        self.x, self.y = x,y
        self.b = batch_size
        self.idx = th.range(0, self.b-1).long()
        self.train = train
        self.augment = augment
        self.sidx = 0

    def __next__(self):
        if self.train:
            self.idx.random_(0,self.n-1)

            x,y  = th.index_select(self.x, 0, self.idx), \
                    th.index_select(self.y, 0, self.idx)

            if self.augment:
                r = np.random.randint(0, 2, self.b)
                x = x.numpy()
                for i in xrange(self.b):
                    if r[i] == 1:
                        x[i] = np.fliplr(x[i])
                    elif r[i] == 2:
                        x[i] = np.flipud(x[i])

                x = th.from_numpy(x)
        else:
            s = self.sidx
            e = min(s+self.b-1, self.n-1)
            #print s,e

            self.idx = th.range(s, e).long()
            self.sidx += self.b
            if self.sidx >= self.n:
                self.sidx = 0

            x,y  = th.index_select(self.x, 0, self.idx), \
                th.index_select(self.y, 0, self.idx)
        return x, y

    next = __next__

    def __iter__(self):
        return self

def mnist(opt):
    d1, d2 = datasets.MNIST('/local2/pratikac/mnist', train=True), \
            datasets.MNIST('/local2/pratikac/mnist', train=False)

    # d1.train_data = (d1.train_data.float() - 126.)/126.
    # d2.test_data = (d2.test_data.float() - 126.)/126.

    train = sampler_t(opt['b'], d1.train_data.view(-1,1,28,28).float(),
        d1.train_labels)
    val = sampler_t(opt['b'], d2.test_data.view(-1,1,28,28).float(),
        d2.test_labels, train=False)
    return train, val, val

def rotmnist(opt):
    loc = '/local2/pratikac/rotmnist/'
    d1 = np.load(loc+'mnist_all_rotation_normalized_float_train_valid.npy')
    d2 = np.load(loc+'mnist_all_rotation_normalized_float_test.npy')

    train = sampler_t(opt['b'], th.from_numpy(d1[:,:-1]).float().view(-1,1,28,28)/255.,
            th.from_numpy(d1[:,-1]).long())
    val = sampler_t(opt['b'], th.from_numpy(d2[:,:-1]).float().view(-1,1,28,28)/255.,
            th.from_numpy(d2[:,-1]).long(), train=False)

    return train, val, val

def cifar10(opt):
    loc = '/local2/pratikac/cifar/preprocessed/'
    d1 = np.load(loc+'train_all.npz')
    d2 = np.load(loc+'test.npz')

    train = sampler_t(opt['b'], th.from_numpy(d1['data']),
                     th.from_numpy(d1['labels']), augment=opt['augment'])
    val = sampler_t(opt['b'], th.from_numpy(d2['data']),
                     th.from_numpy(d2['labels']), train=False)
    return train, val, val

def cifar100(opt):
    loc = '/local2/pratikac/cifar/preprocessed/'
    d1 = np.load(loc+'cifar-100-train_all.npz')
    d2 = np.load(loc+'cifar-100-test.npz')

    train = sampler_t(opt['b'], th.from_numpy(d1['data']),
                     th.from_numpy(d1['labels']))
    val = sampler_t(opt['b'], th.from_numpy(d2['data']),
                     th.from_numpy(d2['labels']), train=False)
    return train, val, val


def imagenet(opt, only_train=False):
    loc = '/local2/pratikac/imagenet'
    bsz, nw = opt['b'], opt['t']

    traindir = os.path.join(loc, 'train')
    valdir = os.path.join(loc, 'val')

    input_transform = [transforms.Scale(256)]
    affine = []

    normalize = [transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]

    if opt['model'].startswith('vgg'):
        bsz = 64
        input_transform = [transforms.Scale(384)]
        normalize = [transforms.Lambda(lambda img: np.array(img) - np.array([123.68, 116.779, 103.939])),
            transforms.Lambda(lambda img: img[:,:,::-1]),    # RGB -> BGR
            transforms.Lambda(lambda pic:
                th.FloatTensor(pic).transpose(0,1).transpose(0,2).contiguous()
            )
        ]

    train_loader = th.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip()] + affine + normalize
            )),
        batch_size=bsz, shuffle=True,
        num_workers=nw, pin_memory=True)

    val_loader = None
    if not only_train:
        val_loader = th.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose(
                input_transform + [transforms.CenterCrop(224)] + affine + normalize)),
            batch_size=bsz, shuffle=False,
            num_workers=nw, pin_memory=True)

    return train_loader, val_loader