#!/bin/bash
OUTPUT_DIR=/mmfs1/gscratch/h2lab/micdun/mimir/data/pythia_dataloading/pile_no_cr/subsets/

python extract_pile_subsets.py /data/pile/train/{00..15}.jsonl \
    --output_dir $OUTPUT_DIR/0 \
    --subsets wikipedia_\(en\)+github