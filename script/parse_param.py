#!/usr/bin/env python

from collections import defaultdict
import os
import os.path as osp

DIR = osp.dirname(osp.realpath(__file__))

def parse(log_file):
    param = {}
    with open(log_file, "r") as f:
        for l in f.readlines():
            if l.find("use-dropout") != -1:
                param = eval(l)
    return param


def main():
    models = ["session3", "vnmt", "cmu", "cmu_vnmt", 
              "mvnmt1", "mvnmt1_avr", "mvnmt1_rnn", "mvnmt1_txt",
              "mvnmt2", "mvnmt2_avr", "mvnmt2_rnn", "mvnmt2_txt"]

    params = {}
    for model in models:
        log_file = osp.abspath(osp.join(DIR, "../", model, "result", "train.log"))
        param = defaultdict(lambda: "")
        if osp.exists(log_file):
            param.update(parse(log_file))
            params[model] = param

    print("{:<10}, {:<10}, {:<10}, {:<10}, {:<10}, {:<10}, {:<10}, {:<10}, {:<10}, {:<15}, {:<12}, {:<30}, {:<30}".format(
        "model","dim","dim_word","dimv","dim_pic","decay-c","lr","maxlen","batchsize","n-words","fine_tuning", "modeldir", "fine_tuning_dir"))
    for model in models:
        p = params[model]
        print("{:<10}, {:<10}, {:<10}, {:<10}, {:<10}, {:<10}, {:<10}, {:<10}, {:<10}, {:<15}, {:<12}, {:<30}, {:<30}".format(
            model,p["dim"][0],p["dim_word"][0],p["dimv"],p["dim_pic"],p["decay-c"][0],p["learning-rate"][0],
            p["maxlen"],p["batchsize"],p["n-words"], p["fine_tuning"], osp.relpath(p["model"][0]), p["fine_tuning_load"]))



if __name__ == "__main__":
    main()

