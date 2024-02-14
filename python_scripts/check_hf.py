from datasets import load_dataset

ds = load_dataset("mimir.py", "pile_cc", split="ngram_7_0.2")
print(ds['member'][0])
ds = load_dataset("mimir.py", "full_pile", split="none")
print(len(ds['member']))
assert len(ds['member']) == 10000
print(ds["nonmember_neighbors"][0])
print(len(ds["member_neighbors"]))
print(ds['member_neighbors'][0][12])
ds = load_dataset("mimir.py", "arxiv", split="ngram_13_0.8")
print(ds["nonmember_neighbors"][1][9])

assert len(ds['member']) == 1000
assert len(ds["nonmember_neighbors"][0]) == 25
