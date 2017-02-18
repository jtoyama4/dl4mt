#!/bin/bash

METEORJAR="../script/meteor-1.5/meteor-1.5.jar"
RESULTDIR="./result"
SPLIT="test"

python ../script/split_data_wrt_length.py result/test_result.merged.detok.txt result/test_result.merged.detok

for i in 5 10 15 20 25 30
do
    SRC=${RESULTDIR}/${SPLIT}_result.merged.detok.$i.txt
    TRUE="../flickr30k/${SPLIT}.norm.ln.${i}.txt"
    if [ ! -f "${RESULTDIR}/${SPLIT}.${i}.txt" ] ; then
        java -Xmx2G -jar $METEORJAR $SRC $TRUE -l de | tee "${RESULTDIR}/${SPLIT}.${i}.meteor.txt"
    fi
done
