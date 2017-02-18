#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import os
import os.path as osp
import sys

from natsort import natsorted

def main(title, outfile):
    meteor = {}

    models = ["session3", "vnmt", "mvnmt1", "mvnmt1_avr", "mvnmt1_rnn", "mvnmt1_txt"]
    #models = ["vnmt", "mvnmt1", "mvnmt1_avr", "mvnmt1_rnn", "mvnmt1_txt"]
    labels = {"session3":"nmt", "vnmt": "vnmt",
             "mvnmt1" : "G", "mvnmt1_avr": "G+O-AVG",
             "mvnmt1_rnn" : "G+O-RNN", "mvnmt1_txt" : "G+O-TXT"}

    for model in models:
        meteor[model] = []

        result_dir = osp.join("../", model, "all_result")

        pat = re.compile(r"val_result\.(\d+)\.merged\.detok\.meteor\.txt")

        for f in natsorted(os.listdir(result_dir)):
            if pat.match(f):
                with open(osp.join(result_dir,f)) as g:
                    meteor[model].append(float(g.readlines()[-1].split(":")[-1].strip())*100)

        plt.plot(meteor[model], label=labels[model])

    plt.legend()
    plt.title(title)
    plt.xlabel("iteration (x 1000)")
    plt.ylabel("METEOR")
    plt.ylim([40,60])
    plt.xlim([0,30])
    plt.savefig(outfile)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="plot meteor graph")
    parser.add_argument("--out", default="meteor_val.eps", help="output file name")
    parser.add_argument("--title", default="Validation METEOR Score", help="graph title")

    args = parser.parse_args()
    main(args.title, args.out)

