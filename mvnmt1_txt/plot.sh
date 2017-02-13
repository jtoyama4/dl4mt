#!/bin/bash

mkdir -p "result"
python ../script/plot_log_klcost.py --out "result/cost.png" --title "MVNMT Type1 RCNN-TXT" result/train.log
python ../script/plot_log_valcost.py --out "result/valcost.png" --title "MVNMT Type1 RCNN-TXT" result/train.log
python ../script/plot_meteor.py --out "result/meteor" --title "MVNMT Type1 RCNN-TXT" all_result
