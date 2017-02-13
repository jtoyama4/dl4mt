#!/bin/bash

SPLIT="val"

if [ $# -ge 1 ] ; then
    SPLIT=$1
fi

echo "${SPLIT}"

cd multeval

./multeval.sh eval \
--refs ../../flickr30k/${SPLIT}.norm.ln.de \
--hyps-baseline ../../session3/result/${SPLIT}_result.merged.detok.txt \
--hyps-sys1 ../../session3/result/${SPLIT}_result.merged.detok.txt \
--hyps-sys2 ../../vnmt/result/${SPLIT}_result.merged.detok.txt \
--hyps-sys3 ../../cmu/result/${SPLIT}_result.merged.detok.txt \
--hyps-sys4 ../../cmu_vnmt/result/${SPLIT}_result.merged.detok.txt \
--hyps-sys5 ../../mvnmt1/result/${SPLIT}_result.merged.detok.txt \
--hyps-sys6 ../../mvnmt1_avr/result/${SPLIT}_result.merged.detok.txt \
--hyps-sys7 ../../mvnmt1_rnn/result/${SPLIT}_result.merged.detok.txt \
--hyps-sys8 ../../mvnmt1_txt/result/${SPLIT}_result.merged.detok.txt \
--hyps-sys9 ../../mvnmt2/result/${SPLIT}_result.merged.detok.txt \
--hyps-sys10 ../../mvnmt2_avr/result/${SPLIT}_result.merged.detok.txt \
--hyps-sys11 ../../mvnmt2_rnn/result/${SPLIT}_result.merged.detok.txt \
--hyps-sys12 ../../mvnmt2_txt/result/${SPLIT}_result.merged.detok.txt \
--meteor.language de | tee multeval_${SPLIT}.txt