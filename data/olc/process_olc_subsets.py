import os
import json
from datasets import load_dataset

def write(file_path, data):
    # open file in write mode
    with open(file_path, "w") as fp:
        for d in data:
            fp.write(json.dumps(d) + "\n")

subset = "sw_github"
samples=100000
dataset = load_dataset("kernelmachine/open-license-corpus", subset, split="train", streaming=True)

if not os.path.exists(subset):
    os.makedirs(subset)

dataset_head = dataset.take(samples)
print(dataset_head)
write(os.path.join(subset, f"sample.jsonl"), list(dataset_head))

# num_shards = 15
# for i in range(num_shards):
#     dataset.shard(num_shards=num_shards, index=i).to_json(os.path.join(subset, f"{i}.jsonl"))
