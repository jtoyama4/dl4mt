#!/bin/bash

mkdir -p result

python train_nmt.py --dim=256 --dim_word=256 --decay_c=0.001 --lr=1.0 --modeldir=nmt_shuffle/ --batchsize=32 2>&1 |  tee result/train.log
