#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

lens = []
sources = []
with open("/share/data/flickr30k/test.en","r") as f:
    for l in f.readlines():
        s = l.strip()
        lens.append(len(s.split(' ')))
        sources.append(s)


idx = np.argsort(lens)[::-1]

for i in idx[:20]:
    print("{:<5}, {:<2}: {}".format(i, lens[i], sources[i]))


plt.hist(lens,bins=7, range=(0,35))
plt.xlabel("Source Sentence Word Length")
plt.ylabel("Number")
plt.savefig("length_hist.eps")
