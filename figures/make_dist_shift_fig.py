#!/usr/bin/env python3
"""Regenerate dist_shift.png (main paper, Sec. "Describing distribution shift in words").

Two panels: top-10 rising and top-10 falling attributes (change in mean
attribute cosine similarity over unlabeled images) of ImageNet-Sketch (left)
and ImageNet-R (right) relative to the ImageNet train set.

Extracted from Distribution_Shift.ipynb, with sign-coded colors (Okabe-Ito
blue for risers, vermillion for fallers) replacing the original per-bar
rainbow palette; the delta is signed data, so a two-color code is the honest
encoding.

Required inputs (re-extract with extract_clip_feature.py if missing; the
repo's data/ symlink to the feature root was removed):
  data/attributes/{pool}_{model}_{dataset}_{split}_sim_ft.pt  attribute similarity features
  data/attributes/{pool}_processed.txt                        attribute names

Source-profile provenance: the notebook computed the ImageNet reference
profile from the FULL train set (1,024,936 x 634), but the restored
{pool}_{model}_train_sim_ft.pt holds only a 3,115-image subset and ranks
different attributes. The default source is therefore
{pool}_{model}_fulltrain_mean_sim_ft.pt, a (1, 634) mean profile verified
against the stored per-image sims (max abs diff 8e-6); regenerate it with
--recompute-source, which needs features/imagenet_RN50/train.pt and
{pool}_{model}_processed_attribute_embd.pt.
"""
import argparse

import pandas as pd
import torch
import matplotlib.pyplot as plt

RISE, FALL = "#0072B2", "#D55E00"  # Okabe-Ito blue / vermillion


def mean_scores(path):
    return torch.load(path).mean(dim=0)


def top_deltas(target_mean, source_mean, attr_names, k=10):
    delta = pd.Series((target_mean - source_mean).numpy(), index=attr_names)
    return pd.concat([delta.nlargest(k), delta.nsmallest(k).iloc[::-1]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="vaw_attributes_simple")
    p.add_argument("--model", default="RN50")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--source-sim", default=None,
                   help="override path to the ImageNet source profile .pt")
    p.add_argument("--recompute-source", action="store_true",
                   help="rebuild the full-train mean profile from raw CLIP "
                        "features and attribute embeddings, then save it to "
                        "the default source path")
    p.add_argument("--target-sims", nargs=2, default=None,
                   help="override paths to [sketch, rendition] sim_ft.pt")
    p.add_argument("--out", default="dist_shift.png")
    args = p.parse_args()

    attr_names = [l.strip() for l in
                  open(f"{args.data_root}/attributes/{args.pool}_processed.txt")]
    src = args.source_sim or (f"{args.data_root}/attributes/"
                              f"{args.pool}_{args.model}_fulltrain_mean_sim_ft.pt")
    if args.recompute_source:
        attr = torch.load(f"{args.data_root}/attributes/"
                          f"{args.pool}_{args.model}_processed_attribute_embd.pt",
                          map_location="cpu").float()
        attr = attr / attr.norm(dim=-1, keepdim=True)
        feat = torch.load(f"{args.data_root}/features/imagenet_{args.model}/train.pt",
                          map_location="cpu")
        sums = torch.zeros(attr.shape[0], dtype=torch.float64)
        for i in range(0, feat.shape[0], 100000):
            f = feat[i:i + 100000].float()
            f = f / f.norm(dim=-1, keepdim=True)
            sums += (100.0 * (f @ attr.T)).sum(0).double()
        torch.save((sums / feat.shape[0]).float().unsqueeze(0), src)
        print(f"recomputed source profile from {feat.shape[0]} images -> {src}")
    tgts = args.target_sims or [
        f"{args.data_root}/attributes/{args.pool}_{args.model}_imS_test_sim_ft.pt",
        f"{args.data_root}/attributes/{args.pool}_{args.model}_imR_test_sim_ft.pt",
    ]

    source_mean = mean_scores(src)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=120)
    for ax, tgt, title in zip(axes, tgts, ["ImageNet-Sketch", "ImageNet-R"]):
        deltas = top_deltas(mean_scores(tgt), source_mean, attr_names)
        colors = [RISE if v > 0 else FALL for v in deltas.values]
        ax.bar(deltas.index, deltas.values, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("delta cosine-similarity")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
