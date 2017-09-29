DATA_DIR=$1
DOWNLOAD_URL="https://miranda.snu.ac.kr/index.php/s/ulW7nko9T0V6mWc/download"
FILENAME=subj
LABELS="subjective objective"
SCRIPT_DIR=$(dirname $0)

bash ./$SCRIPT_DIR/prepare_bc.sh "$DATA_DIR" "$DOWNLOAD_URL" "$FILENAME" $LABELS
