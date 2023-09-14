import argparse
import io
import json
import os
import transformers
import re
import numpy as np
from collections import defaultdict, Counter
from nltk.tokenize import WhitespaceTokenizer

from tqdm import tqdm

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in tqdm(f)] 

def process_texts(data, min_len):
    subset_samples = defaultdict(list)
    subset_counts = Counter()
    for dp in tqdm(data):
        pile_subset = dp["meta"]["pile_set_name"].replace(" ", "_").replace("-", "_").lower()
        subset_counts.update([pile_subset])
        text = dp["text"]
        tokenized = text.split()
        # Initial simple filter to get candidates surpassing min_len requirement
        if len(tokenized) >= min_len:
            subset_samples[pile_subset].append({
                "raw": text
            })

    return subset_samples, subset_counts


def write(file_path, data):
    # open file in write mode
    with open(file_path, "w") as fp:
        for d in data:
            fp.write(json.dumps(d) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_paths', default=[], nargs="*")
    parser.add_argument('--benchmark_dir', type=str, default="./")
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--min_len', type=int, default=100)
    parser.add_argument('--max_len', type=int, default=200)

    args = parser.parse_args()
    print(args)

    output_dir = args.benchmark_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

    tokenizer = WhitespaceTokenizer()
    mask_tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir="/gscratch/h2lab/micdun/mimir/cache_dir")

    print("Loading data")
    data = []
    for path in args.data_paths:
        data += read_jsonl(path)
        
    print(len(data))
    subset_samples, subset_counts = process_texts(data, args.min_len)
    print(subset_counts)
    for subset, samples in subset_samples.items():
        subset_output_dir = os.path.join(output_dir, subset)
        if not os.path.exists(subset_output_dir):
            os.makedirs(subset_output_dir)

        np.random.shuffle(samples)
        truncated_raw_samples = []
        truncated_text_samples = []
        unique_texts = set()
        for sample in samples:
            text = sample["raw"]

            # use nltk tokenizer to split on whtiespace while preserving idx spans per character
            whitespace_tokenized = list(tokenizer.span_tokenize(text))

            # Recheck min_len requirement
            if len(whitespace_tokenized) >= args.min_len:
                # Last span within the 100-200 word requirement
                last_span = whitespace_tokenized[min(args.max_len, len(whitespace_tokenized)) - 1]
                trunc_idx = last_span[1]
                trunc_text = text[:trunc_idx]
                mask_tokenized = mask_tokenizer(trunc_text)["input_ids"]

                # Make sure mask tokenized version of truncated text fits in mask model and isn't a repeat 
                if len(mask_tokenized) <= 512 and trunc_text not in unique_texts:
                    truncated_raw_samples.append({"text": text})
                    truncated_text_samples.append({"text": trunc_text})
                    unique_texts.add(trunc_text)
                    if len(truncated_raw_samples) == args.n_samples:
                        break
        print(f"subset: {subset}, number selected: {len(truncated_raw_samples)}")
        write(os.path.join(subset_output_dir, f"{args.split}_raw.jsonl"), truncated_raw_samples)
        write(os.path.join(subset_output_dir, f"{args.split}_text.jsonl"), truncated_text_samples)
        
