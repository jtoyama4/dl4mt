#!/bin/bash

mkdir -p result

python ./train_mvnmt.py --dim=256 --dim_word=256 --dimv=256 --dim_pic=256 --decay_c=0.001 --lr=1.0 --modeldir=mvnmt2_att/ --batchsize=32 --fine_tuning --finetunedir="nmt" 2>&1 | tee result/train.log

