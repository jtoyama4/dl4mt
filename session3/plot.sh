#!/bin/bash

mkdir -p "result"
python ../script/plot_log_cost.py --out "result/cost.png" --title "NMT" result/train.log
