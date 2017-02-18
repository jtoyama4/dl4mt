#!/bin/bash

mkdir -p "result"
python ../script/plot_log_klcost.py --out "result/cost.eps" --title "VNMT" result/train.log
python ../script/plot_log_valcost.py --out "result/valcost.eps" --title "VNMT" result/train.log
python ../script/plot_meteor.py --out "result/meteor" --title "VNMT" all_result
