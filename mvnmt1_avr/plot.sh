#!/bin/bash

mkdir -p "result"
python ../script/plot_log_klcost.py --out "result/cost.eps" --title "G+O-AVG" result/train.log
python ../script/plot_log_valcost.py --out "result/valcost.eps" --title "G+O-AVG" result/train.log
python ../script/plot_meteor.py --out "result/meteor" --title "G+O-AVR" all_result
