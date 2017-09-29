DATA_DIR=$1
TMP_DIR=/tmp/mrpc
PREPROCESS="sed -f tokenizer.sed"
DOWNLOAD_URL="https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi"

echo $TMP_DIR
mkdir -p $DATA_DIR
mkdir -p $TMP_DIR
curl -Lo $TMP_DIR/mrpc.msi $DOWNLOAD_URL
cabextract $TMP_DIR/mrpc.msi -d $TMP_DIR/mrpc
cat $TMP_DIR/mrpc/_2DEC3DBE877E4DB192D17C0256E90F1D | tr -d $'\r' > $TMP_DIR/train.txt
cat $TMP_DIR/mrpc/_D7B391F9EAFF4B1B8BCE8F21B20B1B61 | tr -d $'\r' > $TMP_DIR/test.txt

for split in train test
do
    FNAME=$TMP_DIR/$split.txt
    mkdir -p $DATA_DIR/$split
    cut -f4 $FNAME | sed '1d' | tqdm | $PREPROCESS > $DATA_DIR/$split/sents1.txt
    cut -f5 $FNAME | sed '1d' | tqdm | $PREPROCESS > $DATA_DIR/$split/sents2.txt
    cut -f1 $FNAME | sed '1d' > $DATA_DIR/$split/labels.txt
done

rm -rf $TMP_DIR
