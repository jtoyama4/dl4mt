#!/bin/bash

OUTDIR="result"

mkdir -p ${OUTDIR}

for split in "val" "test"
do
java -Xmx2G -jar ../script/meteor-1.5/meteor-1.5.jar ${OUTDIR}/${split}_result.merged.detok.txt ../flickr30k/${split}.norm.ln.de -l de | tee ${OUTDIR}/${split}.meteor.txt
java -Xmx2G -jar ../script/meteor-1.5/meteor-1.5.jar ${OUTDIR}/${split}_result.merged.detok.txt ../flickr30k/${split}.norm.ln.de -l de -norm | tee ${OUTDIR}/${split}.norm.meteor.txt
done

for split in "val" "test"
do
    echo "${split} `tail -n1 ${OUTDIR}/${split}.meteor.txt | cut -d ':' -f 2 | sed -e 's/ //g'` `tail -n1 ${OUTDIR}/${split}.norm.meteor.txt | cut -d ':' -f 2 | sed -e 's/ //g'`"
done
