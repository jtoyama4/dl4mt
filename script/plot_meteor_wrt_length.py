#!/usr/bin/env python

# NOTE: run this script in the script dir

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import os
import os.path as osp

from natsort import natsorted

def main(title, outfile):
    meteor = {}

    models = ["session3", "vnmt", "mvnmt1", "mvnmt1_avr", "mvnmt1_rnn", "mvnmt1_txt"]
    labels = {"session3":"nmt", "vnmt": "vnmt",
             "mvnmt1" : "G", "mvnmt1_avr": "G+O-AVG",
             "mvnmt1_rnn" : "G+O-RNN", "mvnmt1_txt" : "G+O-TXT"}
    #X = [5,10,15,20,25,30]
    X = [10,15,20,25,30]

    for model in models:
        meteor[model] = []
        for i in X:
            with open("../{}/result/test.{}.meteor.txt".format(model, i)) as f:
                meteor[model].append(float(f.readlines()[-1].split(":")[-1].strip())*100)

        plt.plot(X, meteor[model], label=labels[model], marker='o')

    plt.legend()
    plt.title(title)
    plt.xticks(X)
    plt.xlabel("Source Sentence Word Length")
    plt.ylabel("METEOR")
    #plt.ylim([0,60])
    plt.savefig(outfile)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="plot meteor graph")
    parser.add_argument("--out", default="meteor_wrt_length.eps", help="output file name")
    parser.add_argument("--title", default="Test METEOR score w.r.t. source word length", help="graph title")

    args = parser.parse_args()
    main(args.title, args.out)

