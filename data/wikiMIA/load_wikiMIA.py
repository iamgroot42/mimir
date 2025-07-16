from datasets import load_dataset
LENGTH = 128
dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{LENGTH}")
dataset =  dataset.rename_column("input", "text")
members = dataset.filter(lambda example: example["label"] == 1)
nonmembers = dataset.filter(lambda example: example["label"] == 0)
members.to_json(f"WikiMIA{LENGTH}_members.jsonl")
nonmembers.to_json(f"WikiMIA{LENGTH}_nonmembers.jsonl")
