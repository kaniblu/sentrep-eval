DATA_DIR=$1
TMP_DIR=/tmp/rt
PREPROCESS="sed -f tokenizer.sed"
DOWNLOAD_URL="https://miranda.snu.ac.kr/index.php/s/65udotY4oGF4DjF/download"

mkdir -p $DATA_DIR
mkdir -p $TMP_DIR
wget $DOWNLOAD_URL -O $TMP_DIR/rt10662.zip
unzip $TMP_DIR/rt10662.zip -d $TMP_DIR
RT_DIR="$(ls -d $TMP_DIR/*/)"

cat "$RT_DIR/rt-polarity.pos" | $PREPROCESS >> $DATA_DIR/sents.txt

for i in $(seq $(cat "$RT_DIR/rt-polarity.pos" | wc -l))
do 
    echo "pos" >> $DATA_DIR/labels.txt
done

cat "$RT_DIR/rt-polarity.neg" | $PREPROCESS >> $DATA_DIR/sents.txt

for i in $(seq $(cat "$RT_DIR/rt-polarity.neg" | wc -l))
do
    echo "neg" >> $DATA_DIR/labels.txt
done

rm -rf $TMP_DIR
