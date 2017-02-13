#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re


def main(log, title, outfile):
    cost = []
    pat = re.compile("Epoch\s+(\d+)\s+Update\s+(\d+)\s+Cost\s+(.*)\s+UD\s+(.*)")

    with open(log, "r") as f:
        for l in f.readlines():
            r = pat.match(l)
            if r is not None:
                cost.append(float(r.group(3)))

    plt.plot(cost, label="cost", color="b")
    plt.legend()
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.ylim([0,100])
    plt.savefig(outfile)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="plot training cost graph from train log")
    parser.add_argument("--out", default="cost.png", help="output file name")
    parser.add_argument("--title", default="NMT", help="graph title")
    parser.add_argument("log", help="log file witch to be parsed")

    args = parser.parse_args()
    main(args.log, args.title, args.out)

