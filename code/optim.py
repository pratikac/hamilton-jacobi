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

class SGD(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.01, momentum=0, dampening=0,
             weight_decay=0, nesterov=True)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(SGD, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None):
        loss = None
        if closure is not None:
            loss = closure()

        c = self.config
        wd = c['weight_decay']
        mom = c['momentum']
        damp = c['dampening']
        nesterov = c['nesterov']
        lr = c['lr']

        for w in self.param_groups[0]['params']:
            dw = w.grad.data

            if wd > 0:
                dw.add_(wd, w.data)
            if mom > 0:
                state = self.state[id(w)]
                if 'mdw' not in state:
                    state['mdw'] = dw.clone()
                buf = state['mdw']
                buf.mul_(mom).add(1-damp, dw)

                if nesterov:
                    dw.add_(mom, buf)
                else:
                    dw = buf

            w.data.add_(-lr, dw)

        return loss

class ESGD(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.01, momentum=0, damp=0,
                 weight_decay=0, nesterov=True,
                 L=0, eps=1e-4, g0=1e-2, g1=0, rho=0,
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

        lr = c['lr']
        rho = c['rho']
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


        if not 't' in state:
            state['t'] = 0
            N = state['N']
            tmp = th.FloatTensor(N).cuda()
            state['wc'] = tmp.clone()
            state['dwc'] = tmp.clone()
            state['dw'] = tmp.clone().zero_()

            state['cache'] = {}
            cache = state['cache']
            cache['w'] = tmp.clone().zero_()
            cache['dw'] = tmp.clone().zero_()
            cache['mw'] = tmp.clone().zero_()

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

        cf = 0
        for i in xrange(L):
            dw.zero_()
            unflatten_params(model, w)
            cf, cerr = closure()
            flatten_params(model, w, dw)

            eta.normal_()
            dw.add_(g, w - state['wc']).add_(eps/np.sqrt(0.5*llr), eta)

            w.add_(-llr, dw)
            mw.mul_(beta1).add_(1-beta1, w)

        dw = state['dw'].zero_()
        if L > 0:
            if rho > 0:
                dw.add_(rho, state['dwc'])
            dw.add_(state['wc'] - mw)
        else:
            dw.add_(state['dwc'])

        if verbose and state['t'] % 100 == 0:
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

class SGLD(Optimizer):
    def __init__(self, params, config = {}):
        defaults = dict(t=0, lr=0.1, momentum=0.9, dampening=0,
                weight_decay=0, nesterov=True, eps=1e-4,
                gamma=0.01, model_bias=False)

        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]
        super(SGLD, self).__init__(params, config)
        self.config = config

    def step(self, closure, model, criterion):
        params = self.param_groups[0]['params']

        c = self.config
        lr, mom, wd, damp, nesterov, eps = c['lr'], c['momentum'], c['weight_decay'], \
                                            c['dampening'], c['nesterov'], c['eps']

        state = self.state
        if not 't' in state:
            state['t'] = 0
            state['wc'], state['mdw'], state['eta'] = None, [], []
            for w in params:
                state['mdw'].append(deepcopy(w.grad.data))
                state['eta'].append(deepcopy(w.grad.data))
            
            if c['model_bias']:
                state['wc'] = deepcopy(self.param_groups[0]['params'])

        if not c['model_bias']:
            state['wc'] = params

        f,err = closure()        
        state['t'] += 1

        debug = dict(w=0, dw=0, eta=0, dF=0, df=0, f=f)

        for wc,w,mdw,eta in zip(state['wc'], params, state['mdw'], state['eta']):
            debug['w'] += th.norm(w).data[0]

            dw = w.grad.data
            debug['df'] += th.norm(dw)

            # add bias term
            dw.add_(c['gamma'], (w-wc).data)
            debug['dF'] += th.norm(wc-w).data[0]

            if wd > 0:
                dw.add_(wd, w.data)
            if mom > 0:
                mdw.mul_(mom).add_(1-damp, dw)
                if nesterov:
                    dw.add_(mom, mdw)
                else:
                    dw = mdw

            debug['dw'] += th.norm(dw)
            
            # add noise
            eta.normal_()
            debug['eta'] += th.norm(eta)*eps/np.sqrt(0.5*lr)

            dw.add_(eps/np.sqrt(0.5*lr), eta)            

            # update weights
            w.data.add_(-lr, dw)

        if c['verbose'] and state['t'] % 100 == 0:
            d = debug
            print   ('t: %04d, f: %.2e, df: %.2e, dF: %.2e, dw: %.2e, eta: %.2e, w: %.2e')% \
                    (state['t']/100, d['f'], d['df'], d['dF'], d['dw'], d['eta'], d['w'])

        return f,err

class HJB(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.01, momentum=0, damp=0,
                 weight_decay=0, nesterov=True,
                 L=0, eps=1e-4, g0=1e-2, g1=0)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(HJB, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for EntropySGDControl, model and criterion'
        assert self.config['L'] > 0, 'L = 0'

        mf,merr = closure()
        
        c = self.config
        state = self.state

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

        params = self.param_groups[0]['params']

        # initialize
        if not 't' in state:
            state['t'] = 0
            state['wc'], state['mdw'] = [], []
            for w in params:
                state['wc'].append(deepcopy(w.data))
                state['mdw'].append(deepcopy(w.grad.data))

            state['sgld'] = dict(mw=deepcopy(state['wc']),
                                    mdw=deepcopy(state['mdw']),
                                    eta=deepcopy(state['mdw']),
                                    lr = 0.1,
                                    beta1 = 0.75)

        lp = state['sgld']
        for i,w in enumerate(params):
            state['wc'][i].copy_(w.data)
            lp['mw'][i].copy_(w.data)
            lp['mdw'][i].zero_()
            lp['eta'][i].normal_()

        state['debug'] = dict(wwpd=0, df=0, dF=0, g=0, eta=0)        
        llr, beta1 = lp['lr'], lp['beta1']
        g = g0*(1+g1)**state['t']

        Mp = 1e3
        for i in xrange(L):
            f,err = closure()

            alpha2 = 0
            copy_into_mw = False
            for wc, w in zip(state['wc'], params):
                alpha2 += th.norm(wc-w.data)**2
            falpha = f + alpha2/2.
            if falpha <= Mp:
                Mp = falpha
                copy_into_mw = True

            for wc,w,mw,mdw,eta in zip(state['wc'], params, \
                                    lp['mw'], lp['mdw'], lp['eta']):
                dw = w.grad.data

                if wd != 0:
                    dw.add_(wd, w.data)
                if mom != 0:
                    mdw.mul_(mom).add_(1-damp, dw)
                    if nesterov:
                        dw.add_(mom, mdw)
                    else:
                        dw = mdw

                # add noise
                eta.normal_()
                dw.add_(-g, wc-w.data).add_(eps/np.sqrt(0.5*llr), eta)

                # update weights
                w.data.add_(-llr, dw)

                if copy_into_mw:
                    mw.copy_(w.data)

        if L > 0:
            # copy model back
            for i,w in enumerate(params):
                w.data.copy_(state['wc'][i])
                w.grad.data.copy_(w.data-lp['mw'][i])   # plug in the grad here

        for w,mdw,mw in zip(params, state['mdw'], lp['mw']):
            dw = w.grad.data

            if wd > 0:
                dw.add_(wd, w.data)
            if mom > 0:
                mdw.mul_(mom).add_(1-damp, dw)
            if nesterov:
                dw.add_(mom, mdw)
            else:
                dw = mdw
            
            w.data.add_(-lr, dw)

        return mf,merr

class PME(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.01, momentum=0, damp=0,
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
        h = np.sqrt(1./g)*dwcn
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

class FB(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.01, momentum=0, damp=0,
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

        m = 2
        Mp = 3

        if not 't' in state:
            state['t'] = 0
            state['wc'] = th.FloatTensor(N).cuda()
            state['dwc'] = th.FloatTensor(N).cuda()
            state['p'] = th.FloatTensor(N).cuda()

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
        p = state['p']
        cache_w = state['cache']['w'].mul_(0)
        cache_dw = state['cache']['dw'].mul_(0)

        # initialize
        state['p'].normal_().mul_(1/np.sqrt(N))*dwcn
        cf = 0

        for i in xrange(int(L/2)):
            w.copy_(state['wc'])
            w.add_(dt, p)
            unflatten_params(model, w)
            cf, cerr = closure()
            flatten_params(model, cache_w, cache_dw)
            p.copy_(cache_dw)

        for i in xrange(int(L/2)):
            w.copy_(state['wc'])
            w.add_(-dt/2., p)
            unflatten_params(model, w)
            cf, cerr = closure()
            flatten_params(model, cache_w, cache_dw)
            p.copy_(cache_dw)

        dw = state['dwc'].zero_()
        dw.add_(p)

        if verbose and state['t'] % 100 == 0:
            debug = dict(dw=dw.norm(), dwc=state['dwc'].norm(),
                dwdwc=th.dot(dw, state['dwc'])/dw.norm()/state['dwc'].norm(),
                f=cf, wc=wcn,
                g=g,
                dt=dt)
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

class ESGDAVG(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.01, momentum=0, damp=0,
                 weight_decay=0, nesterov=True,
                 L=100, g0=1e-2, g1=0,
                 verbose=False)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(ESGDAVG, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for LL, model and criterion'
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

        m = 2
        Mp = 3

        if not 't' in state:
            state['t'] = 0
            state['wc'] = th.FloatTensor(N).cuda()
            state['dwc'] = th.FloatTensor(N).cuda()

            state['cache'] = {}
            state['cache']['w'] = th.FloatTensor(N).cuda().zero_()
            state['cache']['dw'] = th.FloatTensor(N).cuda().zero_()

            state['dw'] = th.FloatTensor(N).cuda().zero_()
            state['mdw'] = th.FloatTensor(N).cuda().zero_()

        state['t'] += 1
        flatten_params(model, state['wc'], state['dwc'])
        wcn, dwcn = state['wc'].norm(), state['dwc'].norm()

        g = g0*(1+g1)**state['t']

        w = state['cache']['w'].zero_()
        cache_dw = state['cache']['dw'].zero_()

        w.copy_(state['wc'])
        dw = state['dw'].zero_()

        cf = 0
        # forward = positive gradient
        for i in xrange(int(L/2)):
            cache_dw.zero_()
            unflatten_params(model, w)
            cf, cerr = closure()
            flatten_params(model, w, cache_dw)
            cache_dw.add_(g, w-state['wc']).mul_((Mp-cf)**(m-1))
            w.add_(-lr, cache_dw)

            dw.add_(cache_dw)

        # w.copy_(state['wc'])
        # for i in xrange(int(L/2)):
        #     cache_dw.zero_()
        #     unflatten_params(model, w)
        #     cf, cerr = closure()
        #     flatten_params(model, w, cache_dw)
        #     cache_dw.mul_(-1)
        #     cache_dw.add_(g, w-state['wc']).mul_((Mp-cf)**(m-1))
        #     w.add_(-lr, cache_dw)

        #     dw.add_(0.05, cache_dw)

        dw.mul_(1./float(L))
        #dw.add_(1-beta*(Mp-mf)**(m-1), state['dwc'])

        if verbose and state['t'] % 100 == 0:
            debug = dict(dw=dw.norm(), dwc=state['dwc'].norm(),
                dwdwc=th.dot(dw, state['dwc'])/dw.norm()/state['dwc'].norm(),
                f=cf,
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

        # update weights
        w = state['wc']
        w.add_(-lr, dw)
        unflatten_params(model, w)
        mf,merr = closure()

        return mf,merr

class LL(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.01, momentum=0, damp=0,
                 weight_decay=0, nesterov=True,
                 L=100, eps=1e-4, g0=1e-2, g1=0,
                 verbose=False)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(LL, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for LL, model and criterion'
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

        if not 't' in state:
            state['t'] = 0
            state['wc'] = th.FloatTensor(N).cuda()
            state['dwc'] = th.FloatTensor(N).cuda()

            state['cache'] = {}
            state['cache']['w'] = th.FloatTensor(N).cuda().zero_()
            state['cache']['dw'] = th.FloatTensor(N).cuda().zero_()

            state['cache']['y'] = th.FloatTensor(N).cuda().zero_()
            state['cache']['z'] = th.FloatTensor(N).cuda().zero_()

            state['dw'] = th.FloatTensor(N).cuda().zero_()
            state['mdw'] = th.FloatTensor(N).cuda().zero_()
            state['eta'] = th.FloatTensor(N).cuda()

        state['t'] += 1
        flatten_params(model, state['wc'], state['dwc'])
        wcn, dwcn = state['wc'].norm(), state['dwc'].norm()

        g = g0*(1+g1)**state['t']

        dw = state['dw'].zero_()
        eta = state['eta']

        llr = 0.1
        cf = 0
        Mm, Mp = -1e3, 1e3

        w = state['cache']['w'].zero_()
        cache_dw = state['cache']['dw'].zero_()
        w.copy_(state['wc'])

        for i in xrange(int(L/2)):
            cache_dw.zero_()
            unflatten_params(model, w)
            cf, cerr = closure()

            copy_in = False
            falpha = cf + (w-state['wc']).norm()**2*g/2.
            if falpha <= Mp:
                Mp = falpha
                copy_in = True

            flatten_params(model, w, cache_dw)
            cache_dw.add_(g, w-state['wc'])

            # add noise
            eta.normal_()
            cache_dw.add_(eps/np.sqrt(0.5*llr), eta)

            w.add_(-llr, cache_dw)

            if copy_in:
                state['cache']['y'].copy_(w)

        w = state['cache']['w'].zero_()
        cache_dw = state['cache']['dw'].zero_()
        w.copy_(state['wc'])

        for i in xrange(int(L/2)):
            cache_dw.zero_()
            unflatten_params(model, w)
            cf, cerr = closure()

            copy_in = False
            falpha = cf - (w-state['wc']).norm()**2*g/2.
            if Mm <= falpha:
                Mm = falpha
                copy_in = True

            flatten_params(model, w, cache_dw)
            cache_dw.add_(2*g, w-state['wc'])

            eta.normal_()
            cache_dw.add_(eps/np.sqrt(0.5*llr), eta)

            w.add_(-llr, cache_dw)

            if copy_in:
                state['cache']['z'].copy_(w)

        # copy grad in
        dw.copy_(state['wc'] - state['cache']['y'])
        dw.add_(1, state['wc'] - state['cache']['z'])

        if verbose and state['t'] % 10 == 0:
            debug = dict(dw=dw.norm(), dwc=state['dwc'].norm(),
                dwdwc=th.dot(dw, state['dwc'])/dw.norm()/state['dwc'].norm(),
                f=cf,
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

        # update weights
        w = state['wc']
        w.add_(-lr, dw)
        unflatten_params(model, w)
        mf,merr = closure()

        return mf,mer

class PMELAP(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.01, momentum=0, damp=0,
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