from datasets import load_dataset

ds = load_dataset("mimir.py", "the_pile_full_pile")
print(ds)
ds = load_dataset("mimir.py", "the_pile_arxiv")
print(ds['member'][0]['text'])
