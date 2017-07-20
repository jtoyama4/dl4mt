#!/bin/bash

mkdir -p result

python train_cmu.py --dim=256 --dim_word=256 --dim_pic=256 --decay_c=0.0005 --lr=1.0 --modeldir=cmu/ --batchsize=32  2>&1 | tee result/train.log
