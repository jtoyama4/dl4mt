
#!/bin/bash

mkdir -p result

python train_vnmt.py --dim=256 --dim_word=256 --dimv=512 --decay_c=0.0001 --lr=1.0 --modeldir=cmu_vnmt/ --batchsize=32 --fine_tuning --finetunedir="nmt" 2>&1 | tee result/train.log
