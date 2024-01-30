"""
    Script to push cache to HuggingFace as a dataset. Uploaded data is public by default.
"""
from simple_parsing import ArgumentParser
import os

from mimir.data_utils import HFCompatibleDataset
from mimir.custom_datasets import load_data
from mimir.utils import get_cache_path


def main(args):
    prefix = f"cache_100_200_{args.num_records}_512"
    target = args.target
    nrgram_suffix = args.ngram_suffix

    for split in ["train", "test"]:
        # Load data
        records = load_data(
            os.path.join(
                get_cache_path(),
                prefix,
                split,
                f"the_pile_{target}{nrgram_suffix}.jsonl",
            )
        )
        # Load neighbors (if they exist)
        neighbors = None
        neighbor_path = os.path.join(
            get_cache_path(),
            prefix,
            f"{split}_neighbors",
            f"the_pile_{target}{nrgram_suffix}_neighbors_25_bert_in_place_swap.jsonl",
        )
        if os.path.exists(neighbor_path):
            neighbors = load_data(neighbor_path)
        ds = HFCompatibleDataset(records, neighbors)
        # TODO: Not familiar with this, but looks like direct ch.Dataset cannot be pushed. Need to figure out the best way to do this.
        # One alternative is to keep 'neighbors' separate and upload them directly via jsonl files
        # ds.push_to_hub(f"iamgroot42/mimir", split=f"{target}{nrgram_suffix}_{split}")


if __name__ == "__main__":
    # Extract relevant configurations from config file
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--target", help="Data source to upload", required=True)
    parser.add_argument("--ngram_suffix", help="Specific ngram_suffix?", default="")
    parser.add_argument(
        "--num_records",
        help="How many records does this source have?",
        default=1000,
        type=int,
    )
    args = parser.parse_args()
    main(args)
