DATA_DIR=$1
DOWNLOAD_URL=$2
FILENAME=$3
ARGS=( "$@" )
LABELS=("${ARGS[@]:3}")
TMP_DIR=/tmp/$FILENAME
PREPROCESS="sed -f tokenizer.sed"

mkdir -p $DATA_DIR
mkdir -p $TMP_DIR
wget $DOWNLOAD_URL -O $TMP_DIR/data.zip
unzip $TMP_DIR/data.zip -d $TMP_DIR
DIR="$(ls -d $TMP_DIR/*/)"

for LABEL in "${LABELS[@]}"
do
    SRC_PATH="$DIR/$FILENAME.$LABEL"
    cat "$SRC_PATH" | $PREPROCESS >> $DATA_DIR/sents.txt

    for i in $(seq $(cat "$SRC_PATH" | wc -l))
    do 
        echo "$LABEL" >> $DATA_DIR/labels.txt
    done
done

rm -rf $TMP_DIR
