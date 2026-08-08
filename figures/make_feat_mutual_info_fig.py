#!/usr/bin/env python3
"""Regenerate Feat_Mutual_Info.png (main paper, Sec. "Structure of the attribute features").

Three panels: pairwise mutual information among (1) CLIP embedding dimensions,
(2) attribute similarity features, (3) attribute features restricted to the
top-5 attributes of three classes.

Extracted from Disentanglement_measure_mutual_info.ipynb with one deliberate
change: the original used sns.diverging_palette(220, 10), a diverging map, on
mutual information, which is unsigned; this script uses the sequential
'viridis' map with vmin=0. Update the figure caption accordingly when the
figure is regenerated (caption currently describes the old teal-to-red map).

Panel-3 attribute lists: the original figure used the probe-selected top-5
attributes recorded in notebook cell 39 (the probe jsons under results/
were not restored). Those lists are hardcoded here as the default;
pass --cls-attri-json to select top-5 from a meanTopk json instead.

Two implementation notes relative to the notebook: the pairwise MI values
are computed in parallel with a fixed random_state (the notebook relied on
the global numpy seed through pd.corr, which is order-dependent), and the
diagonal is set to 1.0 exactly as pd.DataFrame.corr does. The two large MI
matrices are cached in <out>.cache.npz; delete that file to force a
recompute.

Required inputs (re-extract with extract_clip_feature.py if missing):
  data/attributes/{pool}_{model}{shot}_{split}_sim_ft.pt   attribute similarity features
  data/features/{dataset}_{model}/{split}.pt               raw CLIP features
  data/attributes/{pool}_processed.txt                     attribute names
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_regression

CMAP = "viridis"  # sequential; MI has no meaningful midpoint

# Disentanglement_measure_mutual_info.ipynb cell 39 (probe top-5 per class).
ORIGINAL_SEL_ATTRS = {
    "park bench": ["reclining", "leaning back", "relaxing", "kneeling", "swinging"],
    "chocolate syrup": ["chocolate", "buttoned", "sprinkled", "dark brown", "glazed"],
    "hay": ["grassy", "thatched", "wearing hat", "bamboo", "knee high"],
}


def custom_mi_reg(a, b):
    return mutual_info_regression(a.reshape(-1, 1), b.reshape(-1),
                                  random_state=0)[0]


def mi_matrix(df, n_jobs=16):
    cols = df.columns
    X = df.to_numpy()
    n = len(cols)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    vals = Parallel(n_jobs=n_jobs, batch_size=256)(
        delayed(custom_mi_reg)(X[:, i], X[:, j]) for i, j in pairs)
    M = np.eye(n)  # diagonal 1.0, matching pd.DataFrame.corr(method=...)
    for (i, j), v in zip(pairs, vals):
        M[i, j] = M[j, i] = v
    return pd.DataFrame(M, index=cols, columns=cols)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="vaw_attributes_simple")
    p.add_argument("--model", default="RN50")
    p.add_argument("--shot-prefix", default="_1shot")
    p.add_argument("--split", default="train")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--classes", nargs=3, default=["park bench", "chocolate syrup", "hay"],
                   help="three classes for the top-5 panel")
    p.add_argument("--cls-attri-json", default=None,
                   help="optional meanTopk_cls_attri.json to select top-5 from, "
                        "instead of the notebook's original hardcoded lists")
    p.add_argument("--n-jobs", type=int, default=16)
    p.add_argument("--out", default="Feat_Mutual_Info.png")
    args = p.parse_args()

    attr_names = [l.strip() for l in
                  open(f"{args.data_root}/attributes/{args.pool}_processed.txt")]
    sim_ft = torch.load(f"{args.data_root}/attributes/"
                        f"{args.pool}_{args.model}{args.shot_prefix}_{args.split}_sim_ft.pt",
                        map_location="cpu")
    clip_ft = torch.load(f"{args.data_root}/features/"
                         f"imagenet{args.shot_prefix}_{args.model}/{args.split}.pt",
                         map_location="cpu")

    attr_df = pd.DataFrame(data=sim_ft.float().numpy().T, index=attr_names).T
    clip_df = pd.DataFrame(clip_ft.float().numpy())

    if args.cls_attri_json:
        cls_attri = json.load(open(args.cls_attri_json))
        sel = {}
        for c in args.classes:
            ranked = cls_attri[c]
            if isinstance(ranked, dict):  # json stores {"positive": [...]}
                ranked = ranked["positive"]
            sel[c] = ranked[:5]
    else:
        sel = {c: ORIGINAL_SEL_ATTRS[c] for c in args.classes}
    sel_attrs = []
    for c in args.classes:
        for a in sel[c]:
            if a not in sel_attrs:
                sel_attrs.append(a)
    sel_df = attr_df[sel_attrs]

    cache_path = f"{args.out}.cache.npz"
    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        clip_mi = pd.DataFrame(cache["clip"])
        attr_mi = pd.DataFrame(cache["attr"], index=attr_names, columns=attr_names)
        print(f"loaded cached MI matrices from {cache_path}")
    else:
        print("computing CLIP-dimension MI matrix ...")
        clip_mi = mi_matrix(clip_df, args.n_jobs)
        print("computing attribute-feature MI matrix ...")
        attr_mi = mi_matrix(attr_df, args.n_jobs)
        np.savez_compressed(cache_path, clip=clip_mi.to_numpy(),
                            attr=attr_mi.to_numpy())
        print(f"cached MI matrices to {cache_path}")
    sel_mi = mi_matrix(sel_df, args.n_jobs)

    fig, axes = plt.subplots(1, 3, figsize=(30, 8))
    for ax, mi, title in [
        (axes[0], clip_mi, f"CLIP embedding ({clip_ft.shape[1]}-d)"),
        (axes[1], attr_mi, f"Attribute features ({len(attr_names)} attributes)"),
        (axes[2], sel_mi, "Top-5 attributes of " + ", ".join(args.classes)),
    ]:
        sns.heatmap(mi, cmap=CMAP, vmin=0, square=True, ax=ax)
        ax.set_title(title)
    plt.savefig(args.out, bbox_inches="tight", dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
