#!/bin/bash
n=50
trials=5
for subset in "arxiv" "hackernews" # "wikipedia_(en)"
do
    echo generating paraphrases for $subset
    python gen.py \
        "/mmfs1/gscratch/h2lab/micdun/mimir/cache_dir/cache_100_200_1000_512/train/the_pile_${subset}_ngram_13_<0.8_truncated.jsonl" \
        --domain $subset \
        --n $n \
        --trials $trials \
        --output_dir out/gpt4/
done
