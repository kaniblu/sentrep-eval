#!/bin/bash


DATA_DIR=$1
TMP_DIR=/tmp/snli
PREPROCESS="sed -f tokenizer.sed"
DOWNLOAD_URL="https://miranda.snu.ac.kr/index.php/s/hX8dufSFYR3gEru/download"

mkdir -p $DATA_DIR
mkdir -p $TMP_DIR
wget $DOWNLOAD_URL -O $TMP_DIR/snli.zip
unzip -o $TMP_DIR/snli.zip -d $TMP_DIR/snli
DIR=$TMP_DIR/snli

for SPLIT in train dev test
do
    FNAME="$DIR/snli_1.0/snli_1.0_$SPLIT.txt"
    OUT_DIR="$DATA_DIR/$SPLIT"
    TMP_PATH=$TMP_DIR/$SPLIT.txt

    mkdir -p $OUT_DIR

    awk '{ if ( $1 != "-" ) { print $0; } }' $FNAME | cut -f 1,6,7 | sed '1d' > "$TMP_PATH"
    cat "$TMP_PATH" | cut -f1 > "$OUT_DIR/labels.txt"
    cat "$TMP_PATH" | cut -f2 | tqdm | $PREPROCESS > "$OUT_DIR/sents1.txt"
    cat "$TMP_PATH" | cut -f3 | tqdm | $PREPROCESS > "$OUT_DIR/sents2.txt"
done

rm -rf $TMP_DIR
