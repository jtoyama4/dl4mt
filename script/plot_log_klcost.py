#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re


def main(log, title, outfile):
    total_cost = []
    cost = []
    klcost = []
    pat = re.compile(r"Epoch\s+(\d+)\s+Update\s+(\d+)\s+Cost\s+(.*)\s+UD\s+(.*)kl-cost\s+(\d+\.\d+)")

    with open(log, "r") as f:
        for l in f.readlines():
            r = pat.match(l)
            if r is not None:
                tc = float(r.group(3))
                klc = float(r.group(5))
                c = tc - klc
                total_cost.append(tc)
                klcost.append(klc)
                cost.append(c)

    plt.plot(klcost, label="klcost")
    plt.plot(total_cost, label="total_cost")
    plt.plot(cost, label="cost")
    plt.legend()
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel("klcost")
    plt.ylim([0,100])
    plt.savefig(outfile)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="plot training klcost graph from train log")
    parser.add_argument("--out", default="klcost.png", help="output file name")
    parser.add_argument("--title", default="NMT", help="graph title")
    parser.add_argument("log", help="log file witch to be parsed")

    args = parser.parse_args()
    main(args.log, args.title, args.out)

