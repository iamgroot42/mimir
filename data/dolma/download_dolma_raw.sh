#!/bin/bash
DATA_DIR="dolma_raw_pes2o"
PARALLEL_DOWNLOADS="8"
DOLMA_VERSION="v1_5-sample"
SUBSAMPLE_FILE="dolma_${DOLMA_VERSION}_subsample.txt"

mkdir -p "${DATA_DIR}"


cat $SUBSAMPLE_FILE | xargs -n 1 -P "${PARALLEL_DOWNLOADS}" wget -P "$DATA_DIR"