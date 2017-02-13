#!?usr/bin/env python

# NOTE: run this script at the root of this repository

import math
import os
import os.path as osp

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
    return math.ceil(float(val) * 1000)/10.0

def main():
    print("{:<10}, {:<10}, {:<10}, {:<10}, {:<10}".format("", "val", "val(n)", "test", "test(n)"))
    for model in MODELS:
        scores = []
        for split in ["val", "test"]:
            with open(osp.join(model,"result","{}.meteor.txt".format(split)), "r") as f:
                scores.append(f.readlines()[-1].split(":")[-1].strip())
            with open(osp.join(model,"result","{}.norm.meteor.txt".format(split)), "r") as f:
                scores.append(f.readlines()[-1].split(":")[-1].strip())
        print("{:<10}, {:<10}, {:<10}, {:<10}, {:<10}".format(model, *map(roundoff,scores)))



if __name__ == "__main__":
    main()
