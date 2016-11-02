#!/bin/bash

OUTDIR="result"

mkdir -p ${OUTDIR}

for split in "val" "test"
do
python ./translate.py -p 5 -k 12  ../models/mvnmt2_rnn/model_mvnmt_rnn.npz ../flickr30k/bitext.train.en.tok.txt.pkl ../flickr30k/bitext.train.de.tok.bpe.txt.pkl ../flickr30k/bitext.${split}.en.tok.txt ../flickr30k/fc7.${split}.npy ../flickr30k/${split}.imglist.txt  ../flickr30k/${split}.class.txt  ../flickr30k/ ${OUTDIR}/${split}_result.txt
sed "s/@@ //g" ${OUTDIR}/${split}_result.txt > ${OUTDIR}/${split}_result.merged.txt
perl ../script/detokenizer.perl -l de < ${OUTDIR}/${split}_result.merged.txt > ${OUTDIR}/${split}_result.merged.detok.txt
done
