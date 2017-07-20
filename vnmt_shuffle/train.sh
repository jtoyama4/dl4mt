#!/bin/bash

mkdir -p result

#python train_vnmt.py --dim=256 --dim_word=256 --dimv=256 --decay_c=0.001 --lr=1.0 --modeldir=vnmt_shuffle/ --batchsize=32 --fine_tuning --finetunedir="nmt_shuffle" 2>&1 | tee result/train.log
#python train_vnmt.py --dim=256 --dim_word=256 --dimv=512 --decay_c=0.001 --lr=1.0 --modeldir=vnmt_shuffle2/ --batchsize=32 --fine_tuning --finetunedir="nmt_shuffle" 2>&1 | tee result/train2.log
python train_vnmt.py --dim=256 --dim_word=256 --dimv=256 --decay_c=0.001 --lr=1.0 --modeldir=vnmt_shuffle/ --batchsize=32 --fine_tuning --finetunedir="nmt_shuffle" 2>&1 | tee result/train.log

