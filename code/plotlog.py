import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

import os, sys, glob, pdb, re, json
sns.set()

colors = sns.color_palette("husl", 8)

def get_params(s):
    t = s[s.rfind('/')+6:s.find('_opt_')]
    _s = s[s.find('_opt_')+5:-4]
    r = json.loads(_s)
    r = {k: v for k,v in r.items() if k in whitelist}
    r['t'] = t
    return r

def load(dir, expr='*'):
    fs = sorted(glob.glob(dir + '/' + expr + '.log'))
    for i in xrange(len(fs)):
        print i, get_params(fs[i])

    D = [pd.read_csv(f, sep=None, engine='python') for f in fs]
    df = pd.concat(D, keys=[i for i in xrange(len(D))])

    return [get_params(f) for f in fs], df

def loadlog(f, w={}):
    log,summary = [], []
    opt = {}
    for l in open(f):
        r = l.find('| [OPT]')
        if r > -1:
            opt = json.loads(l[r+8:-1])
            print opt
            continue
        r = l.find('| [LOG]')
        if r > -1:
            log.append(json.loads(l[r+8:-1]))
            continue
        r = l.find('| [SUMMARY]')
        if r > -1:
            summary.append(json.loads(l[r+12:-1]))
            continue
    return opt, pd.DataFrame(log), pd.DataFrame(summary)