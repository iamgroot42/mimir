from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm

import os
import json
import numpy as np

def write(file_path, data):
    # open file in write mode
    with open(file_path, "w") as fp:
        for d in data:
            fp.write(json.dumps(d) + "\n")

split = "train"
num_proc = 4
name="v1_5-sample"
# samples = 10
# dataset = load_dataset("allenai/dolma", name=name, split=split, streaming=True)
# dataset_head = dataset.take(samples)
# write(f"dolma-{name}-{split}.jsonl", [{"source": sample["source"], "text": sample["text"]} for sample in dataset_head])

os.environ["DOLMA_DATA_DIR"] = "dolma_raw_temp/"
dataset = load_dataset("dolma", split=split, num_proc=num_proc, cache_dir="./cache")

data_by_source = defaultdict(list)
for sample in tqdm(dataset):
    source = sample["source"]
    text = sample["text"]
    print(sample["created"])
    print(len(text))
    data_by_source[source].append(sample)

for source, samples in data_by_source.items():
    print([s["created"] for s in samples[:100]])