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
num_proc = 8
name="v1_5-sample"
# samples = 10
# dataset = load_dataset("allenai/dolma", name=name, split=split, streaming=True)
# dataset_head = dataset.take(samples)
# write(f"dolma-{name}-{split}.jsonl", [{"source": sample["source"], "text": sample["text"]} for sample in dataset_head])

os.environ["DOLMA_DATA_DIR"] = "dolma_raw_pes2o/"
dataset = load_dataset("dolma", split=split, num_proc=num_proc, cache_dir="./cache")
print(len(dataset))

samples = 100000
idx_sample = np.random.choice(len(dataset), samples, replace=False)
sample_dataset = dataset.select(idx_sample)
data_by_source = defaultdict(list)
for sample in tqdm(sample_dataset):
    source = sample["source"]
    text = sample["text"]
    data_by_source[source].append({
        "source": source,
        "text": text
    })

out_dir = "member_raw_pes2o"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# check lengths make sense
by_source_len = np.sum([len(v)for v in data_by_source.values()])
print(len(sample_dataset))
assert len(sample_dataset) == by_source_len

for source, samples in data_by_source.items():
    write(os.path.join(out_dir, f"dolma-{name}-{source}-{split}.jsonl"), samples)
    
# sample_dataset.to_json(os.path.join(out_dir, f"dolma-{name}-{split}.jsonl"))