#!/bin/bash

if [ $# -lt 3 ] ; then
    echo "Usage: $0 <model_dir> <type> <outdir> [N] [step]"
    exit
fi

#for model in `find $1 -name "model*.iter*.npz" | sort -V`
#do
#    echo $model
#done

MODEL_DIR=$1
TYPE=$2
OUTDIR=$3
N=1000
STEP=1000

P=10 # process number
#K=12 # beam width
K=1 # beam width
SRC_DICT="../flickr30k/bitext.train.en.tok.txt.pkl"
DST_DICT="../flickr30k/bitext.train.de.tok.txt.pkl"
SRC="../flickr30k/bitext.val.en.tok.txt"
OPTION_FILE="$MODEL_DIR/model_$TYPE.npz.pkl"
if [ "$TYPE" == "nmt" ] ; then
    TRANSLATE_SCRIPT="../session3/translate.py"
else
    TRANSLATE_SCRIPT="../$TYPE/translate.py"
fi

if [ $# -ge 4 ] ; then
    N=$4
fi

if [ $# -ge 5 ] ; then
    STEP=$4
fi

if [ -d "$OUTDIR" ] ; then
    mkdir -p $OUTDIR
fi

while true
do
    MODEL="$MODEL_DIR/model_$TYPE.iter$N.npz"
    if [ ! -e $MODEL ] ; then
        exit
    fi
    #echo $MODEL
    
    TARGET="$OUTDIR/result.$TYPE.$N.txt"
    THEANO_FLAGS='device=cpu' python $TRANSLATE_SCRIPT -p $P -k $K -o $OPTION_FILE $MODEL $SRC_DICT $DST_DICT $SRC $TARGET

    N=$((N+STEP))
done

