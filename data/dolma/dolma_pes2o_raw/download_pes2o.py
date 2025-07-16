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

split = "validation"
num_proc = 8
name="v1_5-sample"
# samples = 10
# dataset = load_dataset("allenai/dolma", name=name, split=split, streaming=True)
# dataset_head = dataset.take(samples)
# write(f"dolma-{name}-{split}.jsonl", [{"source": sample["source"], "text": sample["text"]} for sample in dataset_head])

dataset = load_dataset("allenai/peS2o", name="v2", split=split, num_proc=num_proc, cache_dir="./cache")
print(len(dataset))
print(dataset[0])

# cache s2orc and s2ag separately
s2orc =  dataset.filter(lambda example: example['source'].startswith('s2orc'), num_proc=num_proc)
s2ag =  dataset.filter(lambda example: example['source'].startswith('s2ag'), num_proc=num_proc)
print(len(s2orc))
print(len(s2ag))
out_dir = "../member_raw_pes2o"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

n_samples = 100000
data_by_source = defaultdict(list)
for short_source, subset in [('s2orc', s2orc), ('s2ag', s2ag)]:
    print(f"Sampling {short_source}")
    # idx_sample = np.random.choice(len(subset), n_samples, replace=False)
    sample_dataset = subset #.select(idx_sample)
    for sample in tqdm(sample_dataset):
        source = sample["source"]
        text = sample["text"]
        assert source.startswith(short_source)
        data_by_source[short_source].append({
            "source": source,
            "text": text
        })

for source, samples in data_by_source.items():
    write(os.path.join(out_dir, f"dolma-{name}-{source}-{split}.jsonl"), samples)
    
# sample_dataset.to_json(os.path.join(out_dir, f"dolma-{name}-{split}.jsonl"))
