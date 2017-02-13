#!?usr/bin/env python

# NOTE: run this script at the root of this repository

import math
import os
import os.path as osp
import re

MODELS = ["session3","vnmt", "cmu","cmu_vnmt",
          "mvnmt1","mvnmt1_avr","mvnmt1_rnn","mvnmt1_txt",
          "mvnmt2","mvnmt2_avr","mvnmt2_rnn","mvnmt2_txt"]

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def roundoff(val):
    val = float(val)
    if val >= 1.0:
        return val
    return math.ceil(val * 1000)/10.0

def main():

    pat = re.compile(r"system\s+(\d+)\s+(.*)\s+\(.*\)\s+(.*)\s+\(.*\)\s+(.*)\s+\(.*\)\s+(.*)\s+\(.*\)")

    multeval = {}
    for split in ["val", "test"]:
        multeval[split] = {}
        with open("script/multeval/multeval_{}.txt".format(split), "r") as f:
            for l in f.readlines():
                r = pat.match(l)
                if r != None:
                    #print(MODELS[int(r.group(1))-1], r.group(2), r.group(3))
                    multeval[split][MODELS[int(r.group(1))-1]] = {}
                    multeval[split][MODELS[int(r.group(1))-1]]["bleu"] = float(r.group(2))
                    multeval[split][MODELS[int(r.group(1))-1]]["meteor"] = float(r.group(3))

    print("{:<10}, {:<10}, {:<10}, {:<10}, {:<10}, {:<10}, {:<10}, {:<10}, {:<10}".format("", "val", "val(m)", "val(n)", "val(b)", "test", "test(m)", "test(n)", "test(b)"))
    for model in MODELS:
        scores = []
        for split in ["val", "test"]:
            with open(osp.join(model,"result","{}.meteor.txt".format(split)), "r") as f:
                scores.append(f.readlines()[-1].split(":")[-1].strip())
            scores.append(multeval[split][model]["meteor"])
            with open(osp.join(model,"result","{}.norm.meteor.txt".format(split)), "r") as f:
                scores.append(f.readlines()[-1].split(":")[-1].strip())
            scores.append(multeval[split][model]["bleu"])
        print("{:<10}, {:<10}, {:<10}, {:<10}, {:<10}, {:<10}, {:<10} {:<10}, {:<10}".format(model, *map(roundoff,scores)))



if __name__ == "__main__":
    main()
