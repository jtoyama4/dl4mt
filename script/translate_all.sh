#!/bin/bash

if [ $# -lt 3 ] ; then
    echo "Usage: $0 <model> <type> <outdir> [N] [step]"
    exit
fi

#for model in `find $1 -name "model*.iter*.npz" | sort -V`
#do
#    echo $model
#done

MODEL_DIR="../models/$1"
TYPE=$2
OUTDIR=$3
N=1000
STEP=1000

P=3 # process number
#K=12 # beam width
K=12 # beam width
SRC_DICT="../flickr30k/bitext.train.en.tok.txt.pkl"
DST_DICT="../flickr30k/bitext.train.de.tok.txt.pkl"
SRC="../flickr30k/bitext.test.en.tok.txt"
OPTION_FILE="$MODEL_DIR/model_$TYPE.npz.pkl"
if [ "$1" == "nmt" ] ; then
    TRANSLATE_SCRIPT="../session3/translate.py"
else
    TRANSLATE_SCRIPT="../$1/translate.py"
fi

if [ $# -ge 4 ] ; then
    N=$4
fi

if [ $# -ge 5 ] ; then
    STEP=$5
fi

if [ ! -d "$OUTDIR" ] ; then
    mkdir -p $OUTDIR
fi

while true
do
    MODEL="$MODEL_DIR/model_$TYPE.iter$N.npz"
    echo "$MODEL"
    if [ ! -e $MODEL ] ; then
        echo "$MODEL does not exists"
        exit
    fi
    #echo $MODEL
    
    TARGET="$OUTDIR/result.$TYPE.$N.txt"
    THEANO_FLAGS='device=cpu' python $TRANSLATE_SCRIPT -p $P -k $K -o $OPTION_FILE $MODEL $SRC_DICT $DST_DICT $SRC $TARGET

    N=$((N+STEP))
done

