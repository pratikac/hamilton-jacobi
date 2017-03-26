from torch.optim import Optimizer
from copy import deepcopy
import numpy as np
import torch as th
import models

def flatten_params(model, fw, dfw):
    fw.zero_()
    dfw.zero_()
    idx = 0
    for w in model.parameters():
        n = w.numel()
        fw[idx:idx+n].copy_(w.data.view(-1))
        dfw[idx:idx+n].copy_(w.grad.data.view(-1))
        idx += n

def unflatten_params(model, fw):
    idx = 0
    for w in model.parameters():
        w.data.copy_(fw[idx:idx + w.nelement()]).view(w.size())
        idx += w.nelement()

class ESGD(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.1, momentum=0, damp=0,
                 weight_decay=0, nesterov=True,
                 L=0, eps=1e-4, g0=1e-2, g1=0, rho=0,
                 mult=False, hjb=False, sgld=False, ll=False,
                 verbose=False)

        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(ESGD, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for Entropy-SGD, model and criterion'
        mf,merr = closure()

        state = self.state
        c = self.config

        if not 'N' in state:
            state['N'] = models.num_parameters(model)

        hjb = c['hjb']
        sgld = c['sgld']
        ll = c['ll']

        lr = c['lr']
        rho = c['rho']
        mult = c['mult']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['damp']
        nesterov = c['nesterov']
        L = c['L']
        eps = c['eps']
        g0 = c['g0']
        g1 = c['g1']
        verbose = c['verbose']

        llr, beta1 = 0.1, 0.75
        if hjb or ll:
            beta1 = 1e-4

        if not 't' in state:
            state['t'] = 0
            N = state['N']
            tmp = th.FloatTensor(N).cuda()
            state['wc'] = tmp.clone()
            state['dwc'] = tmp.clone()
            state['dw'] = tmp.clone().zero_()

            state['cache'] = {}
            cache = state['cache']
            for k in ['w', 'dw', 'mw', 'mdw', 'y', 'z']:
                state['cache'][k] = tmp.clone().zero_()

            state['eta'] = tmp.clone()
            state['mdw'] = tmp.clone().zero_()

        state['t'] += 1
        flatten_params(model, state['wc'], state['dwc'])

        g = g0*(1+g1)**state['t']

        cache = state['cache']
        w, dw, mw = cache['w'], cache['dw'], cache['mw']
        eta = state['eta']

        w.copy_(state['wc'])
        mw.copy_(state['wc'])

        maxf = 3.0
        cf = 0
        for i in xrange(L):
            dw.zero_()
            unflatten_params(model, w)
            cf, cerr = closure()
            flatten_params(model, w, dw)
            if wd > 0:
                dw.add_(wd, w)

            if ll and i > L/2:
                # reverse quadratic term in LL loop,
                # but compute distance from the right place
                dw.add_(-5*g, w - state['wc'])
            else:
                dw.add_(g, w - state['wc'])

            eta.normal_()
            dw.add_(eps/np.sqrt(0.5*llr), eta)

            if mult:
                dw.mul_((maxf-cf))

            # ascent for the later half
            if ll and i > L/2:
                dw.mul_(-0.1)

            if mom > 0:
                cache['mdw'].mul_(mom).add_(1-damp, dw)
                if nesterov:
                    dw.add_(mom, cache['mdw'])
                else:
                    dw = cache['mdw']

            w.add_(-llr, dw)
            mw.mul_(beta1).add_(1-beta1, w)

            if ll:
                if i <= L/2:
                    cache['y'].copy_(mw)
                else:
                    cache['z'].copy_(mw)

        dw = state['dw'].zero_()
        if ll:
            dw.add_(state['wc'] - cache['y'])
            dw.add_(state['wc'] - cache['z'])
        else:
            if L > 0:
                if rho > 0:
                    dw.add_(rho, state['dwc'])
                dw.add_(state['wc'] - mw)
            else:
                dw.add_(state['dwc'])

        if sgld:
            eta.normal_()
            dw.add_(eps/np.sqrt(0.5*lr), eta)

        if verbose and state['t'] % 25 == 0:
            debug = dict(dw=dw.norm(), dwc=state['dwc'].norm(),
                dwdwc=th.dot(dw, state['dwc'])/dw.norm()/state['dwc'].norm(),
                f=cf, g=g)
            print {k : round(v, 4) for k,v in debug.items()}

        if wd > 0:
            dw.add_(wd, state['wc'])
        if mom > 0:
            state['mdw'].mul_(mom).add_(1-damp, dw)
            if nesterov:
                dw.add_(mom, state['mdw'])
            else:
                dw = state['mdw']

        w = state['wc']
        w.add_(-lr, dw)
        unflatten_params(model, w)
        mf,merr = closure()

        return mf,merr

class SGD(ESGD):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.1, momentum=0.9, dampening=0,
             weight_decay=0, nesterov=True, L=0)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(SGD, self).__init__(params, config)
        self.config = config

class SGLD(ESGD):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.1, momentum=0.9, dampening=0,
            weight_decay=0, nesterov=True, L=0, sgld=True)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(SGLD, self).__init__(params, config)
        self.config = config

class HJB(ESGD):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.1, momentum=0.9, damp=0,
                 weight_decay=0, nesterov=True,
                 L=0, eps=1e-4, g0=1e-2, g1=0,
                 verbose=False,
                 mult=False, hjb=True)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(HJB, self).__init__(params, config)
        self.config = config

class ESGDAVG(ESGD):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.1, momentum=0.9, damp=0,
                 weight_decay=0, nesterov=True,
                 L=100, g0=1e-2, g1=0,
                 verbose=False,
                 mult=True)

        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(ESGDAVG, self).__init__(params, config)
        self.config = config

class FB(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.1, momentum=0, damp=0,
                 weight_decay=0, nesterov=True,
                 L=100, g0=1e-2, g1=0,
                 verbose=False)

        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(FB, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for FB, model and criterion'
        assert self.config['L'] > 0, 'L = 0'

        mf,merr = closure()

        state = self.state
        c = self.config

        if not 'N' in state:
            state['N'] = models.num_parameters(model)

        lr = c['lr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['damp']
        nesterov = c['nesterov']
        L = c['L']
        g0 = c['g0']
        g1 = c['g1']
        N = state['N']
        verbose = c['verbose']

        if not 't' in state:
            state['t'] = 0
            state['wc'] = th.FloatTensor(N).cuda()
            state['dwc'] = th.FloatTensor(N).cuda()
            state['p'] = th.FloatTensor(N).cuda()
            state['dw'] = th.FloatTensor(N).cuda()

            state['cache'] = {}
            state['cache']['w'] = th.FloatTensor(N).cuda().zero_()
            state['cache']['dw'] = th.FloatTensor(N).cuda().zero_()

            state['mdw'] = th.FloatTensor(N).cuda().zero_()

        state['t'] += 1
        flatten_params(model, state['wc'], state['dwc'])
        wcn, dwcn = state['wc'].norm(), state['dwc'].norm()

        g = g0*(1+g1)**state['t']
        dt = g

        w = state['cache']['w']
        cache_w = state['cache']['w'].mul_(0)
        cache_dw = state['cache']['dw'].mul_(0)

        state['p'].normal_().mul_(1/np.sqrt(N))*dwcn
        p = state['p']

        llr = 0.1
        cf = 0
        pn1, pn2 = 0, 0
        for i in xrange(int(L/2)):
            w.copy_(state['wc'])
            w.add_(dt, p)
            unflatten_params(model, w)

            cache_dw.zero_()
            cf, cerr = closure()
            flatten_params(model, cache_w, cache_dw)
            if wd > 0:
                cache_dw.add_(wd, cache_w)
            p.add_(llr, cache_dw)
            pn1 = p.norm()

        for i in xrange(int(L/2)):
            w.copy_(state['wc'])
            w.add_(-dt/2., p)
            unflatten_params(model, w)

            cache_dw.zero_()
            cf, cerr = closure()
            flatten_params(model, cache_w, cache_dw)
            if wd > 0:
                cache_dw.add_(wd, cache_w)
            p.add_(llr, cache_dw)
            pn2 = p.norm()

        dw = state['dw'].zero_()
        dw.add_(p)

        if verbose and state['t'] % 100 == 0:
            debug = dict(dw=dw.norm(), dwc=state['dwc'].norm(),
                dwdwc=th.dot(dw, state['dwc'])/dw.norm()/state['dwc'].norm(),
                f=cf, wc=wcn,
                g=g,
                dt=dt,
                pn1=pn1, pn2=pn2)
            print {k : round(v, 4) for k,v in debug.items()}

        if wd > 0:
            dw.add_(wd, state['wc'])
        if mom > 0:
            state['mdw'].mul_(mom).add_(1-damp, dw)
            if nesterov:
                dw.add_(mom, state['mdw'])
            else:
                dw = state['mdw']

        # update weights
        w = state['wc']
        w.add_(-lr, dw)
        unflatten_params(model, w)
        mf,merr = closure()

        return mf,merr

class LL(ESGD):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.1, momentum=0.9, damp=0,
                 weight_decay=0, nesterov=True,
                 L=100, eps=1e-4, g0=1e-2, g1=0,
                 verbose=False,
                 ll=True)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(LL, self).__init__(params, config)
        self.config = config

class PME(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.1, momentum=0, damp=0,
                 weight_decay=0, nesterov=True,
                 L=100, g0=1e-2, g1=0,
                 verbose=False)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(PME, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for PME, model and criterion'
        assert self.config['L'] > 0, 'L = 0'

        mf,merr = closure()

        state = self.state
        c = self.config

        if not 'N' in state:
            state['N'] = models.num_parameters(model)

        lr = c['lr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['damp']
        nesterov = c['nesterov']
        L = c['L']
        eps = c['eps']
        g0 = c['g0']
        g1 = c['g1']
        N = state['N']
        verbose = c['verbose']

        m = 2
        Mp = 3

        # initialize
        if not 't' in state:
            state['t'] = 0
            state['wc'] = th.FloatTensor(N).cuda()
            state['dwc'] = th.FloatTensor(N).cuda()
            state['dw'] = th.FloatTensor(N).cuda()

            state['cache'] = {}
            state['cache']['w'] = th.FloatTensor(N).cuda().zero_()
            state['cache']['dw'] = th.FloatTensor(N).cuda().zero_()
            state['cache']['eta'] = th.FloatTensor(N).cuda()

            state['mdw'] = th.FloatTensor(N).cuda().zero_()

        state['t'] += 1
        flatten_params(model, state['wc'], state['dwc'])
        wcn, dwcn = state['wc'].norm(), state['dwc'].norm()

        g = g0*(1+g1)**state['t']
        h = np.sqrt(g)*dwcn
        #dt = 1
        #beta = L*dt/h**2      # this is the discretization
        beta = 0.5

        w = state['cache']['w']
        dw = state['dw'].mul_(0)
        cache_dw = state['cache']['dw'].mul_(0)
        cf = 0
        for i in xrange(L):
            w.copy_(state['wc'])

            r = state['cache']['eta'].normal_().mul_(1/np.sqrt(N))
            w.add_(h, r)
            unflatten_params(model, w)
            cf, cerr = closure()
            flatten_params(model, w, cache_dw)

            dw.add_(beta/float(L)*(Mp-cf)**(m-1), cache_dw)

        dw.add_(1 - beta*(Mp-mf)**(m-1), state['dwc'])

        if verbose and state['t'] % 100 == 0:
            debug = dict(dw=dw.norm(), dwc=state['dwc'].norm(),
                dwdwc=th.dot(dw, state['dwc'])/dw.norm()/state['dwc'].norm(),
                f=cf, wc=wcn,
                beta=beta,
                g=g,
                h=h)
            print {k : round(v, 4) for k,v in debug.items()}

        if wd > 0:
            dw.add_(wd, state['wc'])
        if mom > 0:
            state['mdw'].mul_(mom).add_(1-damp, dw)
            if nesterov:
                dw.add_(mom, state['mdw'])
            else:
                dw = state['mdw']

        w = state['wc']
        w.add_(-lr, dw)
        unflatten_params(model, w)
        mf,merr = closure()

        return mf,merr

class PMELAP(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.1, momentum=0, damp=0,
                 weight_decay=0, nesterov=True,
                 L=100, eps=1e-4, g0=1e-2, g1=0,
                 verbose=False)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(PMELAP, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for PMELAP, model and criterion'
        assert self.config['L'] > 0, 'L = 0'

        mf,merr = closure()

        state = self.state
        c = self.config

        if not 'N' in state:
            state['N'] = models.num_parameters(model)

        lr = c['lr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['damp']
        nesterov = c['nesterov']
        L = c['L']
        g0 = c['g0']
        g1 = c['g1']
        eps = c['eps']
        N = state['N']
        verbose = c['verbose']

        m = 2
        Mp = 3

        if not 't' in state:
            state['t'] = 0
            tmp = th.FloatTensor(N).cuda()
            state['wc'] = tmp.clone()
            state['dwc'] = tmp.clone()

            state['cache'] = {}
            cache = state['cache']
            cache['f'], cache['fm'], cache['fmm'] = 0, 0, 0

            for s in ['w', 'dw', 'wm', 'wmm', 'dwm', 'dwmm']:
                cache[s] = tmp.clone().zero_()

            state['dw'] = tmp.clone().zero_()
            state['eta'] = tmp.clone()
            state['mdw'] = tmp.clone().zero_()

        state['t'] += 1
        flatten_params(model, state['wc'], state['dwc'])

        g = g0*(1+g1)**state['t']

        cache = state['cache']

        w, wm, wmm = cache['w'], cache['wm'], cache['wmm']
        dw, dwm, dwmm = cache['dw'], cache['dwm'], cache['dwmm']

        eta = state['eta']

        w.copy_(state['wc'])
        wm.copy_(state['wc'])
        wmm.copy_(state['wc'])

        dw.copy_(state['dwc'])
        dwm.copy_(state['dwc'])
        dwmm.copy_(state['dwc'])

        state['dw'].zero_()
        llr = 0.1
        for i in xrange(L+3):

            # update the cache
            dwmm.copy_(dwm)
            dwm.copy_(dw)
            wmm.copy_(wm)
            wm.copy_(w)
            cache['fmm'] = cache['fm']
            cache['fm'] = cache['f']

            dw.zero_()
            unflatten_params(model, w)
            cache['f'], cerr = closure()
            flatten_params(model, w, dw)

            if i > 3:
                h = 1. #(w-wmm).norm()/2        (cancel with dt)
                state['dw'].add_(1/h**2*(Mp-cache['f'])**(m-1), dw-dwm)
                state['dw'].add_(-1/h**2*(Mp-cache['fm'])**(m-1), dwm-dwmm)

            eta.normal_()
            dw.add_(-g, w-state['wc']).add_(eps/np.sqrt(0.5*llr), eta)
            w.add_(-llr, dw)

        dw = state['dw']
        dw.mul_(1./float(L))
        dw.add_(state['dwc'])

        if verbose and state['t'] % 100 == 0:
            debug = dict(dw=dw.norm(), dwc=state['dwc'].norm(),
                dwdwc=th.dot(dw, state['dwc'])/dw.norm()/state['dwc'].norm(),
                f=cache['f'], fm=cache['fm'], fmm=cache['fmm'],
                g=g)
            print {k : round(v, 4) for k,v in debug.items()}

        if wd > 0:
            dw.add_(wd, state['wc'])
        if mom > 0:
            state['mdw'].mul_(mom).add_(1-damp, dw)
            if nesterov:
                dw.add_(mom, state['mdw'])
            else:
                dw = state['mdw']

        w = state['wc']
        w.add_(-lr, dw)
        unflatten_params(model, w)
        mf,merr = closure()

        return mf,merr