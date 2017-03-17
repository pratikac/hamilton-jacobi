from torch.optim import Optimizer
from copy import deepcopy
import numpy as np
import torch as th

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

class EntropySGD(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.01, momentum=0, damp=0,
                 weight_decay=0, nesterov=True,
                 L=0, eps=1e-4, g0=1e-2, g1=0)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(EntropySGD, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for Entropy-SGD, model and criterion'
        mf,merr = closure()
        
        c = self.config
        lr = c['lr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['damp']
        nesterov = c['nesterov']
        L = c['L']
        eps = c['eps']
        g0 = c['g0']
        g1 = c['g1']

        # only deal with the basic group?
        params = self.param_groups[0]['params']

        state = self.state
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

        for i in xrange(L):
            f,_ = closure()
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
                mw.mul_(beta1).add_(1-beta1, w.data)

        if L > 0:
            # copy model back
            for i,w in enumerate(params):
                w.data.copy_(state['wc'][i])
                w.grad.data.copy_(w.data-lp['mw'][i])

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


class EntropySGDControl(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.01, momentum=0, damp=0,
                 weight_decay=0, nesterov=True,
                 L=0, eps=1e-4, g0=1e-2, g1=0)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(EntropySGDControl, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for EntropySGDControl, model and criterion'
        mf,merr = closure()

        c = self.config
        lr = c['lr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['damp']
        nesterov = c['nesterov']
        L = c['L']
        eps = c['eps']
        g0 = c['g0']
        g1 = c['g1']

        # only deal with the basic group?
        params = self.param_groups[0]['params']

        state = self.state
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

        minf = -1e3
        for i in xrange(L):
            f,err = closure()

            alpha = 0
            copy_into_mw = False
            for wc, w in zip(state['wc'], params):
                alpha += th.norm(wc-w.data)
            fpalpha = f + alpha
            if minf < fpalpha:
                minf = fpalpha
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

def flatten_params(model):
    return th.cat([param.data.view(-1) for param in model.parameters()], 0), \
        th.cat([param.grad.data.view(-1) for param in model.parameters()], 0)

def unflatten_params(model, flattened):
    offset = 0
    for param in model.parameters():
        param.data.copy_(flattened[offset:offset + param.nelement()]).view(param.size())
        offset += param.nelement()

class SGDPME(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.01, momentum=0, damp=0,
                 weight_decay=0, nesterov=True,
                 L=100, g0=1e-2, g1=0,
                 verbose=False)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(SGDPME, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for SGDPME, model and criterion'
        mf,merr = closure()

        c = self.config
        lr = c['lr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['damp']
        nesterov = c['nesterov']
        L = c['L']
        eps = c['eps']
        g0 = c['g0']
        g1 = c['g1']
        verbose = c['verbose']

        m = 2
        maxf = 3

        # only deal with the basic group?
        params = self.param_groups[0]['params']
        fw, fdw = flatten_params(model)
        N = fw.numel()

        state = self.state
        # initialize
        if not 't' in state:
            state['t'] = 0
            state['wc'] = deepcopy(fw)
            state['dwc'] = deepcopy(fdw)
            state['mdw'] = deepcopy(fdw)*0
            state['eta'] = deepcopy(fdw)
            state['dw'] = deepcopy(fdw)*0

        state['t'] += 1
        state['wc'].copy_(fw)
        state['dwc'].copy_(fdw)
        wcn, dwcn = state['wc'].norm(), state['dwc'].norm()

        g = g0*(1+g1)**state['t']
        h = np.sqrt(1./g)*dwcn
        #dt = 1
        #beta = L*dt/h**2      # this is the discretization
        beta = 0.5

        dw = state['dw'].mul_(0)
        cf = 0
        for i in xrange(L):
            fw.copy_(state['wc'])

            r = state['eta'].normal_().mul_(1/np.sqrt(N))
            fw.add_(h, r)
            unflatten_params(model, fw)
            cf, cerr = closure()
            _, cdw = flatten_params(model)

            dw.add_(beta/float(L)*(maxf-cf)**(m-1), cdw)

        dw.add_(1 - beta*(maxf-mf)**(m-1), state['dwc'])

        if verbose and state['t'] % 100 == 0:
            debug = dict(dw=dw.norm(), dwc=state['dwc'].norm(),
                dwdwc=th.dot(dw, state['dwc'])/dw.norm()/state['dwc'].norm(),
                f=cf, wc=wcn,
                beta=beta,
                g=g,
                h=h)
            print debug

        if wd > 0:
            dw.add_(wd, state['wc'])
        if mom > 0:
            state['mdw'].mul_(mom).add_(1-damp, dw)
            if nesterov:
                dw.add_(mom, state['mdw'])
            else:
                dw = state['mdw']

        # update weights
        fw = state['wc']

        fw.add_(-lr, dw)
        unflatten_params(model, fw)
        mf,merr = closure()

        return mf,merr

class SGDFB(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.01, momentum=0, damp=0,
                 weight_decay=0, nesterov=True,
                 L=100, g0=1e-2, g1=0,
                 verbose=False)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(SGDFB, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for SGDPME, model and criterion'
        mf,merr = closure()

        c = self.config
        lr = c['lr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['damp']
        nesterov = c['nesterov']
        L = c['L']
        g0 = c['g0']
        g1 = c['g1']
        verbose = c['verbose']

        m = 2
        maxf = 3

        # only deal with the basic group?
        params = self.param_groups[0]['params']
        fw, fdw = flatten_params(model)
        N = fw.numel()

        state = self.state
        # initialize
        if not 't' in state:
            state['t'] = 0
            state['wc'] = deepcopy(fw)
            state['dwc'] = deepcopy(fdw)
            state['mdw'] = deepcopy(fdw)*0
            state['p'] = deepcopy(fdw)*0

        state['t'] += 1
        state['wc'].copy_(fw)
        state['dwc'].copy_(fdw)
        wcn, dwcn = state['wc'].norm(), state['dwc'].norm()

        g = g0*(1+g1)**state['t']
        dt = g
        llr = 0.1

        state['p'].normal_().mul_(1/np.sqrt(N))*dwcn
        cf = 0
        p = state['p']
        dw = fdw.zero_()
        for i in xrange(int(L/2)):
            fw.copy_(state['wc'])
            fw.add_(dt, p)
            unflatten_params(model, fw)
            cf, cerr = closure()
            _, cdw = flatten_params(model)
            p.add_(llr, cdw)

        for i in xrange(int(L/2)): 
            fw.copy_(state['wc'])
            fw.add_(-dt/2., p)
            unflatten_params(model, fw)
            cf, cerr = closure()
            _, cdw = flatten_params(model)
            p.add_(llr, cdw)

        dw.zero_().add_(p)

        if verbose and state['t'] % 100 == 0:
            debug = dict(dw=dw.norm(), dwc=state['dwc'].norm(),
                dwdwc=th.dot(dw, state['dwc'])/dw.norm()/state['dwc'].norm(),
                f=cf, wc=wcn,
                g=g,
                dt=dt)
            print debug

        if wd > 0:
            dw.add_(wd, state['wc'])
        if mom > 0:
            state['mdw'].mul_(mom).add_(1-damp, dw)
            if nesterov:
                dw.add_(mom, state['mdw'])
            else:
                dw = state['mdw']

        # update weights
        fw = state['wc']

        fw.add_(-lr, dw)
        unflatten_params(model, fw)
        mf,merr = closure()

        return mf,merr