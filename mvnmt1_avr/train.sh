#!/bin/bash

mkdir -p result

python ./train_mvnmt.py --dim=256 --dim_word=256 --dimv=512 --dim_pic=256 --decay_c=0.0005 --lr=1.0 --modeldir=mvnmt1_avr/ --batchsize=32 --fine_tuning --finetunedir="vnmt" 2>&1 | tee result/train.log
