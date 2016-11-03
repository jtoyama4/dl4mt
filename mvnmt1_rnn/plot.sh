#!/bin/bash

mkdir -p "result"
python ../script/plot_log_klcost.py --out "result/cost.png" --title "M1VNMT Typ1 RCNN-RNN" result/train.log
