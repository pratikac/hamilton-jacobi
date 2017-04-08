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
            default=42)
parser.add_argument('--thj',
            type=float,
            help='HJ time',
            default=0.2)
parser.add_argument('--thjnv',
            type=float,
            help='non-viscous HJ time',
            default=0.1)
parser.add_argument('--fp',
            type=float,
            help='FPK viscosity',
            default=0.1)
parser.add_argument('--tfp',
            type=float,
            help='FPK time',
            default=10)
parser.add_argument('-s',
            help='save figures',
            action='store_true')
opt = vars(parser.parse_args())

np.random.seed(opt['seed'])

if opt['s']:
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

    fsz = 24
    plt.rc('font', size=fsz)
    plt.rc('axes', titlesize=fsz*0.8)
    plt.rc('axes', labelsize=fsz)
    plt.rc('xtick', labelsize=fsz)
    plt.rc('ytick', labelsize=fsz)
    plt.rc('legend', fontsize=fsz*0.5)
    plt.rc('figure', titlesize=fsz)

N, M = 500, 3
x = np.linspace(-M, M, N)
dx = x[1] - x[0]

g = stats.norm.pdf(x, 0, 2*M/200.)
g = g/np.sum(g)
f2 = np.convolve(5*(np.random.random(x.shape) - 0.5), g, 'same')
f = x**2 + \
    1.5*np.sin(2*np.pi*x) + f2
# (x**2-2)*(np.abs(x) < 0.5).astype(int) + \
# ((x-0.5)**2-2)*(np.abs(x-0.5) < 0.5).astype(int) + \

r0m, r0v = 2.5, 0.02
t1 = np.exp(-(x-r0m)**2/r0v)*(np.abs(x-r0m) <= 3*np.sqrt(r0v)).astype(int)
r0 = t1/np.sum(t1)/dx

def neumann(z):
    return np.array([z[0]]+z[:-1].tolist()), \
        np.array(z[1:].tolist()+[z[-1]])
def diff(z):
    zl, zr = neumann(z)
    return (zr-zl)/(2*dx)
def diff2(z):
    zl, zr = neumann(z)
    t1 = np.minimum(zl-z, zr-z)/dx
    return np.minimum(0, t1)**2

def lap1d(z):
    zl, zr = neumann(z)
    return (zr+zl - 2*z)/(dx**2)

def hj(u0, r0, a=(1,1), fp=opt['fp'], T=0.2, TFP=20, n=10):

    if not (T > 0):
        return [0], [u0], [r0]

    # u_t  = -0.5*a[1]*|ux|^2 + 0.5*a[0]*uxx
    u0m = np.max(u0)
    dt = 0.5*np.min([dx**2/(u0m*(a[1]+1e-12)),
                    0.5*dx**2/(a[0]+1e-12),
                    0.001])
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

    print '[FPK]'
    # r_t = div(grad u . r) + fp*lap r
    dt = np.abs(ts[1] - ts[0])
    ds = np.min([dx**2/(2*fp), \
                dx/(2*np.max(np.abs(us)))])
    ns = int((TFP/float(n))/ds)
    print 'ds: %.6f, ns: %d'%(ds, ns)

    def stepfpk(r, u, ds=ds):
        return r + ds*(diff(r*diff(u)) + fp*lap1d(r))

    rs = [r0]
    for i in xrange(1, n):
        u1, u2 = us[i-1], us[i]
        r = rs[-1].copy()
        for j in xrange(ns+1):
            u = (u1*(ns-j) + u2*(j))/float(ns)
            r = stepfpk(r, u)
        rs.append(r)

    for i in xrange(len(rs)):
        rs[i] = rs[i]/np.sum(rs[i])/dx
    return ts, us[::-1], rs

print '\n[HJ]'
tsv, vs, rvs = hj(f, r0, (1,1), T=opt['thj'], TFP=opt['tfp'])

print '\n[Burgers]'
ts, nvs, rnvs = hj(f, r0, (0, 1),
         T=opt['thjnv'], TFP=opt['tfp'])

print '\n[SGD]'
ts, sgds, rsgds = hj(f, r0, (1e-12, 1e-12),
        T=opt['thj'], TFP=opt['tfp'], fp=0.2)

plt.figure(1, figsize=(8,7))
plt.clf()
plt.plot(x,f,'k-',lw=1, label=r'$f(x)$')

plt.fill_between(x, f, f+r0, color='grey', alpha='0.35')
plt.fill_between(x, f, f+rsgds[-1]/4., color='grey', alpha='0.9')

plt.plot(x, vs[-1],'indianred',lw=1.5, label=r'$u_{\textrm{viscous\ HJ}}(x,T)$')
plt.fill_between(x, vs[-1], vs[-1]+rvs[-1], color='indianred', alpha='0.75')

plt.plot(x, nvs[-1],'royalblue',lw=1.5, label=r'$u_{\textrm{non-viscous\ HJ}}(x,T)$')
plt.fill_between(x, nvs[-1], nvs[-1]+rnvs[-1]/3.,
        color='royalblue', alpha='0.75')

plt.xticks([])
plt.yticks([])
plt.title(r'Viscous vs. non-viscous Hamilton-Jacobi equation smoothing')
plt.legend(loc='upper center')
if opt['s']:
    plt.savefig('../fig/smoothing.pdf', bbox_inches='tight')