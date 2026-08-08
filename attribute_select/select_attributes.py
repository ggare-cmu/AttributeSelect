"""Select per-class attributes from image data (paper Sec. 4).

Two pairing estimators for the class-attribute score w[c, j]:
  mean   the mean similarity feature over the class's images (no training;
         used for distribution-level applications)
  probe  weights (plus bias) of a linear probe trained on the similarity
         features (used for the classification experiments)

Output JSON:
  {class_name: {"positive": [attr, ...], "values": [w, ...]}}
ranked by descending score, truncated to --topk (0 keeps the full ranking).
"""
import argparse
import json

import torch

from .encode_attributes import load_pool


def subsample_shots(labels, shots, seed=0):
    """Return indices keeping at most `shots` images per class."""
    g = torch.Generator().manual_seed(seed)
    keep = []
    for c in labels.unique(sorted=True):
        idx = torch.where(labels == c)[0]
        idx = idx[torch.randperm(len(idx), generator=g)][:shots]
        keep.append(idx)
    return torch.cat(keep)


def mean_pairing(sim, labels, num_classes):
    w = torch.zeros(num_classes, sim.shape[1])
    for c in range(num_classes):
        w[c] = sim[labels == c].mean(dim=0)
    return w


def probe_pairing(sim, labels, num_classes, epochs=10, lr=1e-4, wd=0.01,
                  batch_size=4096, device="cpu"):
    probe = torch.nn.Linear(sim.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    ds = torch.utils.data.TensorDataset(sim, labels)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(probe(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
    with torch.no_grad():
        return (probe.weight + probe.bias.unsqueeze(1)).cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim-features", required=True, help=".pt from attribute_features")
    p.add_argument("--labels", required=True, help="<split>_label.pt from extract_features")
    p.add_argument("--classes", required=True, help="<split>_classes.txt from extract_features")
    p.add_argument("--pool", required=True, help="attribute pool txt (same order as features)")
    p.add_argument("--pairing", choices=["mean", "probe"], default="probe")
    p.add_argument("--topk", type=int, default=5, help="attributes kept per class; 0 = full ranking")
    p.add_argument("--shots", type=int, default=0, help="images per class used for selection; 0 = all")
    p.add_argument("--seed", type=int, default=0, help="shot-subsampling seed")
    p.add_argument("--out", required=True, help="output JSON path")
    args = p.parse_args()

    sim = torch.load(args.sim_features, map_location="cpu").float()
    labels = torch.load(args.labels, map_location="cpu")
    classes = [l.strip() for l in open(args.classes) if l.strip()]
    pool = load_pool(args.pool)
    assert sim.shape[1] == len(pool), "pool and similarity features disagree"

    if args.shots:
        keep = subsample_shots(labels, args.shots, args.seed)
        sim, labels = sim[keep], labels[keep]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.pairing == "mean":
        w = mean_pairing(sim, labels, len(classes))
    else:
        w = probe_pairing(sim, labels, len(classes), device=device)

    k = args.topk or w.shape[1]
    values, idx = torch.topk(w, k=k, dim=1)
    selection = {
        cls: {"positive": [pool[j] for j in idx[c]],
              "values": [round(v, 4) for v in values[c].tolist()]}
        for c, cls in enumerate(classes)
    }
    with open(args.out, "w") as f:
        json.dump(selection, f, indent=1)
    print(f"selected top-{k} attributes for {len(classes)} classes "
          f"({args.pairing} pairing, {args.shots or 'all'} shots) -> {args.out}")


if __name__ == "__main__":
    main()
