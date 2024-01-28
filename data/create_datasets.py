import argparse
import datasets
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

def process_texts(data, min_len, provided_subset=None):
    subset_samples = defaultdict(list)
    subset_counts = Counter()
    for dp in tqdm(data):
        if provided_subset is not None:
            pile_subset = provided_subset
        elif "meta" in dp:
            pile_subset = dp["meta"]["pile_set_name"].replace(" ", "_").replace("-", "_").lower()

        subset_counts.update([pile_subset])
        text = dp["text"]
        tokenized = text.split()
        # Initial simple filter to get candidates surpassing min_len requirement
        if len(tokenized) >= min_len:
            # TODO: for temporal_wiki, need to append title metadata to front. Should refactor this
            dp["raw"] = dp["title"] + "\n\n" + text if "title" in dp and provided_subset == 'temporal_wiki' else text
            subset_samples[pile_subset].append(dp)

    return subset_samples, subset_counts


def write(file_path, data):
    # open file in write mode
    with open(file_path, "w") as fp:
        for d in data:
            fp.write(json.dumps(d) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_paths', default=[], nargs="*")
    parser.add_argument('--ngram_metadata', action='store_true', help="If filtering on ngram overlap and ngram overlap metadata is provided per sample")
    parser.add_argument('--benchmark_dir', type=str, default="./")
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--min_len', type=int, default=100)
    parser.add_argument('--max_len', type=int, default=200)
    parser.add_argument('--min_ngram_overlap', type=float, default=0.0)
    parser.add_argument('--max_ngram_overlap', type=float, default=0.2)
    parser.add_argument('--provided_subset', type=str, default=None)
    parser.add_argument('--full_doc', action='store_true', help='Chunk the document into equal length substrings')
    parser.add_argument('--tokenize', action='store_true', help='Pretokenize data into npy file')

    args = parser.parse_args()
    print(args)

    n_samples = args.n_samples * 10 if args.tokenize else args.n_samples

    output_dir = args.benchmark_dir
    if args.ngram_metadata:
        output_dir = os.path.join(output_dir, f"{args.min_ngram_overlap}-{args.max_ngram_overlap}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

    tokenizer = WhitespaceTokenizer()
    mask_tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir="/gscratch/h2lab/micdun/mimir/cache_dir")

    print("Loading data")
    data = []
    if args.ngram_metadata:
        print(f"filtering out data with ngram overlap in range [{args.min_ngram_overlap}, {args.max_ngram_overlap})")
        # TODO: just assuming 2 shard files are passed in for now
        shard_0_path, shard_1_path = args.data_paths[0], args.data_paths[1]
        shard_0 = datasets.load_dataset("json", data_files=shard_0_path, split="train")
        shard_1 = datasets.load_dataset("json", data_files=shard_1_path, split="train")
        assert shard_0["original"][0] == shard_1["original"][0] and shard_0["original"][1] == shard_1["original"][1]
        ngram_inclusion = [np.array(in0[:200]) | np.array(in1[:200]) for in0, in1 in tqdm(zip(shard_0["ngram_inclusion"], shard_1["ngram_inclusion"]))]
        ngram_inclusion_full = [np.array(in0) | np.array(in1) for in0, in1 in tqdm(zip(shard_0["ngram_inclusion"], shard_1["ngram_inclusion"]))]
        data = [
            {"text": text, "overlap": np.mean(d), "full_overlap": np.mean(d_full)} for text, d, d_full in tqdm(zip(shard_0["original"], ngram_inclusion, ngram_inclusion_full)) if np.mean(d) < args.max_ngram_overlap and np.mean(d) >= args.min_ngram_overlap
        ] if args.provided_subset else [
            {"text": text, "meta": meta, "overlap": np.mean(d), "full_overlap": np.mean(d_full)} for text, meta, d, d_full in tqdm(zip(shard_0["original"], shard_0["meta"], ngram_inclusion, ngram_inclusion_full)) if np.mean(d) < args.max_ngram_overlap and np.mean(d) >= args.min_ngram_overlap
        ]
        
    else:
        for path in args.data_paths:
            data += read_jsonl(path)
        
    print(len(data))
    subset_samples, subset_counts = process_texts(data, args.min_len, args.provided_subset)
    print(subset_counts)
    for subset, samples in tqdm(subset_samples.items()):
        print("curr subset:", subset)
        subset_output_dir = os.path.join(output_dir, subset)
        if not os.path.exists(subset_output_dir):
            os.makedirs(subset_output_dir)

        np.random.shuffle(samples)
        truncated_raw_samples = []
        truncated_text_samples = []
        unique_texts = set()
        for sample in tqdm(samples):
            text = sample["raw"]

            # use nltk tokenizer to split on whtiespace while preserving idx spans per character
            whitespace_tokenized = list(tokenizer.span_tokenize(text))
            # Recheck min_len requirement
            if not args.full_doc:
                if len(whitespace_tokenized) >= args.min_len:
                    # Last span within the 100-200 word requirement
                    last_span = whitespace_tokenized[min(args.max_len, len(whitespace_tokenized)) - 1]
                    trunc_idx = last_span[1]
                    trunc_text = text[:trunc_idx]
                    mask_tokenized = mask_tokenizer(trunc_text)["input_ids"]

                    # Make sure mask tokenized version of truncated text fits in mask model and isn't a repeat 
                    if len(mask_tokenized) <= 512 and trunc_text not in unique_texts:
                        truncated_raw_samples.append({"text": text})

                        sample["text"] = trunc_text
                        truncated_text_samples.append(sample)
                        unique_texts.add(trunc_text)
                        if len(truncated_raw_samples) == n_samples:
                            break
            else:
                # Get docs that have at least max_len * 2 words
                if len(whitespace_tokenized) >= args.max_len * 2:
                    print("using full doc")
                    # Split doc into as many disjoint max_len samples as possible
                    # window size: max len, stride: max_len // 2
                    window_size = args.max_len
                    stride = window_size
                    substr_samples = []
                    for i in range(0, len(whitespace_tokenized), stride):
                        # stop iterating if sample from start idx is too small
                        if len(whitespace_tokenized) - i < args.min_len:
                            break
                        split_sample = whitespace_tokenized[i:i+window_size]
                        start_span = split_sample[0]
                        end_span = split_sample[-1]
                        start_trunc_idx = start_span[0]
                        end_trunc_idx = end_span[1]
                        trunc_text = text[start_trunc_idx:end_trunc_idx]
                        mask_tokenized = mask_tokenizer(trunc_text)["input_ids"]

                        # Make sure mask tokenized version of truncated text fits in mask model and isn't a repeat 
                        if len(mask_tokenized) <= 512 and trunc_text not in unique_texts:
                            substr_samples.append(trunc_text)
                            unique_texts.add(trunc_text)

                    print("num substrs:", len(substr_samples))
                    if len(substr_samples) > 0:
                        truncated_raw_samples.append({"text": text})
                        truncated_text_samples.append({"substr_samples": substr_samples})
                        if len(truncated_raw_samples) == n_samples:
                            break
        print(f"subset: {subset}, number selected: {len(truncated_raw_samples)}")
        if args.tokenize and not args.full_doc:
            # Tokenize the raw samples and select the first 300 tokens per sample
            # Use the pythia tokenizer
            tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/pythia-12b-deduped", cache_dir="/gscratch/h2lab/micdun/mimir/cache_dir/")
            sample_tokens = []
            for sample in tqdm(truncated_raw_samples, desc="Tokenizing samples"):
                tokens = tokenizer(sample["text"]).input_ids
                if len(tokens) >= 300:
                    sample_tokens.append(tokens[:300])

            print(len(sample_tokens))
            assert len(sample_tokens) >= args.n_samples
            sample_tokens_np = np.array(sample_tokens)
            sample_indices = np.random.choice(sample_tokens_np.shape[0], args.n_samples, replace=False)
            sample_tokens_np = sample_tokens_np[sample_indices,:]
            print(sample_tokens_np.shape)
            np.save(os.path.join(subset_output_dir, f"{args.split}_tk.npy"), sample_tokens_np)

        write(os.path.join(subset_output_dir, f"{args.split}_raw.jsonl"), truncated_raw_samples)
        write(os.path.join(subset_output_dir, f"{args.split}_text.jsonl"), truncated_text_samples)
        
