#!/bin/bash

#if [ $# -lt 2 ] ; then
#    echo "Usage: $0 [N] [step]"
#    exit
#fi

METEORJAR="../script/meteor-1.5/meteor-1.5.jar"
RESULTDIR="./all_result"

TRUE="../flickr30k/bitext.test.de.tok.txt"


for SRC in `find $RESULTDIR -name "test_result.*.merged.detok.txt" | sort -V`
do
    FILE=`basename $SRC .txt`
    NUM=${FILE##test_result.*.merged.detok.txt}

    #echo "${RESULTDIR}/${NUM}.meteor.txt"
    if [ ! -f "${RESULTDIR}/${NUM}.meteor.txt" ] ; then
        java -Xmx2G -jar $METEORJAR $SRC $TRUE -l de | tee "${RESULTDIR}/${NUM}.meteor.txt"
    fi
    if [ ! -f "${RESULTDIR}/${NUM}.norm.meteor.txt" ] ; then
    java -Xmx2G -jar $METEORJAR $SRC $TRUE -l de -norm | tee "${RESULTDIR}/${NUM}.norm.meteor.txt"
    fi
done


