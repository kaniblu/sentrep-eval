DATA_DIR=$1
DOWNLOAD_URL="https://miranda.snu.ac.kr/index.php/s/28g4GvHCnfr8bM2/download"
FILENAME=mpqa
LABELS="pos neg"
SCRIPT_DIR=$(dirname $0)

bash ./$SCRIPT_DIR/prepare_bc.sh "$DATA_DIR" "$DOWNLOAD_URL" "$FILENAME" $LABELS
