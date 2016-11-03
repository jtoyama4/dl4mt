#!/bin/bash

mkdir -p "result"
python ../script/plot_log_klcost.py --out "result/cost.png" --title "CMU-VNMT" result/train.log
python ../script/plot_log_valcost.py --out "result/valcost.png" --title "CMU-VNMT" result/train.log
