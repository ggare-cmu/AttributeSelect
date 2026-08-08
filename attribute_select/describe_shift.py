"""Describe the shift between two image collections in words (paper Sec. 6.1).

Subtracts the source dataset's mean attribute profile from the target's and
reports the attributes that rose and fell the most. Uses unlabeled images
only; no class information is involved.
"""
import argparse

import torch

from .encode_attributes import load_pool


def mean_profile(sim_path):
    return torch.load(sim_path, map_location="cpu").float().mean(dim=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-sim", required=True,
                   help="similarity features .pt of the reference dataset")
    p.add_argument("--target-sim", required=True,
                   help="similarity features .pt of the shifted dataset")
    p.add_argument("--pool", required=True, help="attribute pool txt")
    p.add_argument("--topk", type=int, default=10)
    args = p.parse_args()

    pool = load_pool(args.pool)
    delta = mean_profile(args.target_sim) - mean_profile(args.source_sim)

    rise_v, rise_i = torch.topk(delta, args.topk)
    fall_v, fall_i = torch.topk(-delta, args.topk)
    print(f"top-{args.topk} rising attributes (more present in target):")
    for v, i in zip(rise_v, rise_i):
        print(f"  +{v:.3f}  {pool[i]}")
    print(f"top-{args.topk} falling attributes (less present in target):")
    for v, i in zip(fall_v, fall_i):
        print(f"  -{v:.3f}  {pool[i]}")


if __name__ == "__main__":
    main()
