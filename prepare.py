"""
One-time data preparation for the speedrun.

Downloads pre-tokenized (GPT-2) FineWeb10B shards into ./data/fineweb10B/.
The val shard (shard 0) is always downloaded; pass an integer arg to download
N additional train shards (each ~100M tokens). Default: 9 train shards
(~900M tokens, enough for a full training run).

Usage:
    python prepare.py            # val + 9 train shards (~900M tokens)
    python prepare.py 20         # val + 20 train shards (~2B tokens)
    python prepare.py 103        # full fineweb10B (~10B tokens)

The train/val token streams are part of the fixed speedrun rules and must
NOT be modified by the agent.
"""

import os
import sys

from huggingface_hub import hf_hub_download

LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fineweb10B")
HF_REPO_ID = "kjj0/fineweb10B-gpt2"


def get(fname: str) -> None:
    target = os.path.join(LOCAL_DIR, fname)
    if os.path.exists(target):
        return
    print(f"  Downloading {fname}")
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=fname,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
    )


if __name__ == "__main__":
    num_chunks = int(sys.argv[1]) if len(sys.argv) >= 2 else 9
    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"Cache directory: {LOCAL_DIR}")
    print(f"Downloading val shard + {num_chunks} train shards (~{num_chunks * 100}M train tokens)")
    get("fineweb_val_%06d.bin" % 0)
    for i in range(1, num_chunks + 1):
        get("fineweb_train_%06d.bin" % i)
    print("Done.")
