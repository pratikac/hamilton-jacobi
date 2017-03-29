import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

import os, sys, glob2, pdb, re, json, argparse
import cPickle as pickle
sns.set()

colors = sns.color_palette("husl", 8)

from processlog import *

parser = argparse.ArgumentParser(description='Plotter')
parser.add_argument('-m',
            help='mnistfc | lenet | allcnn', type=str,
            default='mnistfc')
parser.add_argument('-l',
            help='location', type=str,
            default='/Users/pratik/Dropbox/siap17data')
parser.add_argument('-f',
            help='reprocess data',
            action='store_true')
opt = vars(parser.parse_args())

d = loaddir(os.path.join(opt['l'], opt['m']), force=opt['f'])

d = d[(d['summary'] == True) & (d['val'] == True)]
d = d.filter(items=['optim', 'top1', 'L', 'e', 's'])
d.loc[d.L==0,'L'] = 1
d.loc[:,'e'] += 1
d['ee'] = d['e']*d['L']

plt.figure(1)
plt.clf()
sns.tsplot(time='ee',value='top1',data=d,
            unit='s',condition='optim')
plt.title(opt['m'])