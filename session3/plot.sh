#!/bin/bash

mkdir -p "result"
python ../script/plot_log_cost.py --out "result/cost.eps" --title "NMT" result/train.log
python ../script/plot_log_valcost.py --out "result/valcost.eps" --title "NMT" result/train.log
python ../script/plot_meteor.py --out "result/meteor.eps" --title "NMT" all_result
