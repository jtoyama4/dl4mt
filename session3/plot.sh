#!/bin/bash

mkdir -p "result"
python ../script/plot_log_cost.py --out "result/cost.png" --title "NMT" result/train.log
python ../script/plot_log_valcost.py --out "result/valcost.png" --title "NMT" result/train.log
python ../script/plot_meteor.py --out "result/meteor" --title "NMt" all_result
