#!/bin/bash

if [ $# -lt 2 ] ; then
    echo "Usage: $0 <meteojar> <result_dir> <type> [N] [step]"
    exit
fi

METEORJAR=$1
RESULTDIR=$2
TYPE=$3


TRUE="../flickr30k/bitext.val.de.tok.txt"


for SRC in `find $RESULTDIR -name "result.$TYPE.*.txt" | sort -V`
do
    #$echo "java -Xmx2G -jar $METEORJAR $SRC $TRUE -l de | grep 'Final score' | cut -d: -f2 | sed 's/ //g'"
    FILE=`basename $SRC .txt`
    NUM=${FILE##*.}
    echo -n "$NUM "
    java -Xmx2G -jar $METEORJAR $SRC $TRUE -l de | grep 'Final score' | cut -d: -f2 | sed 's/ //g'
done


