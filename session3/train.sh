#!/bin/bash

python train_nmt.py --dim=256 --dim_word=256 --decay_c=0.001 --lr=1.0 --modeldir=nmt/ --batchsize=32 2>&1 | tee train_nmt.log
