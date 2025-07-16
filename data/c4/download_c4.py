from datasets import load_dataset
import json

def write(file_path, data):
    # open file in write mode
    with open(file_path, "w") as fp:
        for d in data:
            fp.write(json.dumps(d) + "\n")

split="validation"
samples = 100000
dataset = load_dataset("c4", "en", split=split, streaming=True)
dataset_head = dataset.take(samples)
write(f"c4-en-{split}.jsonl", list(dataset_head))