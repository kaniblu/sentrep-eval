DATA_DIR=$1
DOWNLOAD_URL="https://miranda.snu.ac.kr/index.php/s/VZhw1WeCHhzkFs2/download"
FILENAME=custrev
LABELS="pos neg"
SCRIPT_DIR=$(dirname $0)

bash ./$SCRIPT_DIR/prepare_bc.sh "$DATA_DIR" "$DOWNLOAD_URL" "$FILENAME" $LABELS
