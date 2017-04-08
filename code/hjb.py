import numpy as np
import os, pdb, sys
import matplotlib.pyplot as plt
import os, sys, glob, pdb, argparse
import cPickle as pickle
import seaborn as sns

import scipy.stats as stats

sns.set_style('ticks')
sns.set_color_codes()

parser = argparse.ArgumentParser(description='HJB simulation')
parser.add_argument('--seed',
            type=int,
            help='seed',
            default=35)
parser.add_argument('--thj',
            type=float,
            help='HJ time',
            default=0)
parser.add_argument('--thjnv',
            type=float,
            help='non-viscous HJ time',
            default=0)
parser.add_argument('--tfp',
            type=float,
            help='FPK time',
            default=0)
parser.add_argument('-s',
            help='save figures',
            action='store_true')
opt = vars(parser.parse_args())

np.random.seed(opt['seed'])

if opt['s']:
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)


N, M = 500, 3
x = np.linspace(-M, M, N)
dx = x[1] - x[0]

g = stats.norm.pdf(x, 0, 2*M/200.)
g = g/np.sum(g)
f2 = np.convolve(5*(np.random.random(x.shape) - 0.5), g, 'same')
f = x**2 + (x**2-2)*(np.abs(x) < 0.5).astype(int) + \
    np.sin(2*np.pi*x) + f2

r0m, r0v = 2.25, 0.1
t1 = np.exp(-(x-r0m)**2/r0v)*(np.abs(x-r0m) <= 3*np.sqrt(r0v)).astype(int)
r0 = t1/np.sum(t1)/dx

def neumann(z):
    return np.array([z[0]]+z[:-1].tolist()), \
        np.array(z[1:].tolist()+[z[-1]])

def diff2(z):
    zl, zr = neumann(z)
    t1 = np.minimum(zl-z, zr-z)/dx
    return np.minimum(0, t1)**2

def lap1d(z):
    zl, zr = neumann(z)
    return (zr+zl - 2*z)/(dx**2)

def hj(u0, r0, a=(2,2), fp=0.2, T=0.2, TFP=20, n=10):

    if not (T > 0):
        return [0], [u0], [r0]

    print 'Starting HJ'
    # u_t  = -0.5*a[1]*|ux|^2 + 0.5*a[0]*uxx

    u0m = np.max(u0)
    dt = 0.5*np.min([dx**2/(u0m*(a[1]+1e-6)), 0.5*dx**2/(a[0]+1e-6)])
    ts = np.arange(0,T,dt)
    nt = len(ts)
    print 'dt: %.6f, nt: %d'%(dt, nt)

    def stephj(u, dt=dt):
        return u + dt*(a[0]/2.*lap1d(u) - a[1]/2.*diff2(u))

    us = [u0]
    for i in xrange(1,nt):
        us.append(stephj(us[i-1]))

    ts, us = ts[::nt//n], us[::-1][::nt//n]

    if not (TFP > 0):
        return ts, us[::-1], [r0]

    print 'Starting FPK'
    # r_t = div(grad u . r) + fp*lap r
    dt = np.abs(ts[1] - ts[0])
    ds = np.min([dx**2/(2*fp), \
                dx/(2*np.max(np.abs(us)))])
    ns = int((TFP/float(n))/ds)
    print 'ds: %.6f, ns: %d'%(ds, ns)

    def stepfpk(r, u, ds=ds):
        t1 = r*u
        tl, tr = neumann(t1)
        dvp = (tr-tl)/(2*dx)
        return r + ds*(dvp + fp*lap1d(r))

    rs = [r0]
    for i in xrange(1, n):
        u1, u2 = us[i-1], us[i]
        r = rs[-1].copy()
        for j in xrange(ns):
            u = (u1*(ns-j) + u2*(j))/float(ns)
            r = stepfpk(r, u)
        rs.append(r)

    for i in xrange(len(rs)):
        rs[i] = rs[i]/np.sum(rs[i])/dx
    return ts, us[::-1], rs

tsv, vs, rvs = hj(f, r0, (1,1), T=opt['thj'], TFP=opt['tfp'])
ts, nvs, rnvs = hj(f, r0, (1e-12, 1),
        T=opt['thjnv'], TFP=opt['tfp'])

plt.figure(1)
plt.clf()
plt.plot(x,f,'k-',lw=1)
plt.plot(x,r0,'k-',lw=1)

plt.plot(x, vs[-1],'r-',lw=1)
plt.plot(x, rvs[-1],'r-',lw=1)

plt.plot(x, nvs[-1],'b-',lw=1)
plt.plot(x, rnvs[-1],'b-',lw=1)