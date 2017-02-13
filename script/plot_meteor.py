#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import os
import os.path as osp

from natsort import natsorted

def main(result_dir, title, outfile):
    meteor = []

    pat = re.compile(r"val_result\.(\d+)\.merged\.detok\.meteor\.txt")

    for f in natsorted(os.listdir(result_dir)):
        if pat.match(f):
            with open(osp.join(result_dir,f)) as g:
                meteor.append(float(g.readlines()[-1].split(":")[-1].strip())*100)

    plt.plot(meteor, label="METEOR", color="b")
    plt.legend()
    plt.title(title)
    plt.xlabel("iteration (x 1000)")
    plt.ylabel("METEOR")
    plt.ylim([0,60])
    plt.savefig(outfile)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="plot meteor graph")
    parser.add_argument("--out", default="meteor.png", help="output file name")
    parser.add_argument("--title", default="NMT", help="graph title")
    parser.add_argument("result_dir", help="result directory witch contain METEOR evaluation result txts")

    args = parser.parse_args()
    main(args.result_dir, args.title, args.out)

