#!/bin/bash

if [ $# -lt 2 ] ; then
    echo "Usage: $0 <meteojar> <result_dir> <type> [N] [step] [max]"
    exit
fi

METEORJAR=$1
RESULTDIR=$2
TYPE=$3


TRUE="../flickr30k/val.norm.ln.de"


for SRC in `find $RESULTDIR -name "result.$TYPE.*.txt" | sort -V`
do
    if [[ "$SRC" =~ result\.$TYPE\.[0-9]+\.txt  ]]; then
        A="`dirname $SRC`/`basename $SRC .txt`"
        if [ -f "$A.no_bpe.txt" ]; then
            continue
        fi
    else 
        continue
    fi
    # merge compound words
    NOBPE="`dirname $SRC`/`basename $SRC .txt`.no_bpe.txt"
    #echo "sed 's/@@ //g' $SRC > $NOBPE"
    sed "s/@@ //g" $SRC > $NOBPE

    # detokenize
    DETOK="`dirname $SRC`/`basename $NOBPE .txt`.detok.txt"
    #echo "perl ../script/detokenizer.perl -l de < $NOBPE > $DETOK"
    perl ../script/detokenizer.perl -l de < $NOBPE > $DETOK 2>/dev/null

    #$echo "java -Xmx2G -jar $METEORJAR $SRC $TRUE -l de | grep 'Final score' | cut -d: -f2 | sed 's/ //g'"
    FILE=`basename $SRC .txt`
    NUM=${FILE##*.}
    echo  "$NUM "\
    `java -Xmx2G -jar $METEORJAR $DETOK $TRUE -l de 2>/dev/null | grep 'Final score' | cut -d: -f2 | sed 's/ //g'`" "\
    `java -Xmx2G -jar $METEORJAR $DETOK $TRUE -l de -norm 2>/dev/null | grep 'Final score' | cut -d: -f2 | sed 's/ //g'`" "\
    #echo "java -Xmx2G -jar $METEORJAR $SRC $TRUE -l de | grep 'Final score' | cut -d: -f2 | sed 's/ //g'"
done


