#!/bin/bash

mkdir -p "result"
python ../script/plot_log_klcost.py --out "result/cost.png" --title "MVNMT Type1 FC7" result/train.log
python ../script/plot_log_valcost.py --out "result/valcost.png" --title "MVNMT Type1 FC7" result/train.log
python ../script/plot_meteor.py --out "result/meteor" --title "MVNMT Type1 FC7" all_result
