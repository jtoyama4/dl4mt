#!/bin/bash

#if [ $# -lt 2 ] ; then
#    echo "Usage: $0 [N] [step]"
#    exit
#fi

#for model in `find $1 -name "model*.iter*.npz" | sort -V`
#do
#    echo $model
#done

MODEL_DIR="../models/cmu_vnmt/"
TYPE="vnmt_cmu"
OUTDIR="all_result"
N=1000
STEP=1000

P=3 # process number
K=12 # beam width
SRC_DICT="../flickr30k/bitext.train.en.tok.txt.pkl"
DST_DICT="../flickr30k/bitext.train.de.tok.bpe.txt.pkl"
SRC="../flickr30k/bitext.test.en.tok.txt"
FC="../flickr30k/fc7.test.npy"
IMGLIST="../flickr30k/test.imglist.txt"
CLASSTXT="../flickr30k/test.class.txt"
IMGBASEDIR="../flickr30k"
OPTION_FILE="$MODEL_DIR/model_$TYPE.npz.pkl"
TRANSLATE_SCRIPT="./translate.py"

if [ $# -ge 1 ] ; then
    N=$1
fi

if [ $# -ge 2 ] ; then
    STEP=$2
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
    
    TARGET="$OUTDIR/test_result.$N.txt"
    TARGET_MERGED="$OUTDIR/test_result.$N.merged.txt"
    TARGET_DETOK="$OUTDIR/test_result.$N.merged.detok.txt"

    if [ ! -f $TARGET ] ; then
        THEANO_FLAGS='device=cpu' python $TRANSLATE_SCRIPT -p $P -k $K -o $OPTION_FILE $MODEL $SRC_DICT $DST_DICT $SRC $FC $IMGLIST $CLASSTXT $IMGBASEDIR $TARGET
    fi
    if [ ! -f ${TARGET_MERGED} ] ; then
        sed "s/@@ //g" ${TARGET} > ${TARGET_MERGED}
    fi
    if [ ! -f ${TARGET_DETOK} ] ; then
        perl ../script/detokenizer.perl -l de < ${TARGET_MERGED} > ${TARGET_DETOK}
    fi

    N=$((N+STEP))
done

