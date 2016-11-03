#!/bin/bash

mkdir -p "result"
python ../script/plot_log_klcost.py --out "result/cost.png" --title "MVNMT Type2 RCNN-RNN" result/train.log
