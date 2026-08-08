"""Zero-shot evaluation with selected attributes (paper Eq. 2).

A test image embedding v is assigned to the class maximizing
  (1/|A_c|) sum_{j in A_c} omega[c, j] * cos(v, t_j),
with omega = 1 (default) or the stored selection values (--weighted).

Protocols:
  attribute-only  each attribute is encoded bare ("colorless")
  classname       each attribute is appended to the class name in the style
                  of Menon & Vondrick ("strawberry, which is colorless")
The classname protocol needs --model to re-encode the composed prompts;
attribute-only can reuse precomputed attribute embeddings via --attribute-embeddings.
"""
import argparse
import json

import torch

from .encode_attributes import encode_texts, load_pool


def compose_prompt(cls, attr):
    cls = cls.replace("_", " ")
    if attr.split(" ", 1)[0] in {"has", "have", "with"}:
        return f"{cls}, which {attr}"
    return f"{cls}, which is {attr}"


def class_prototypes(selection, classes, embed_of, weighted):
    """Per class: mean (optionally weighted) of its attribute prompt embeddings."""
    protos = []
    for cls in classes:
        entry = selection[cls]
        e = torch.stack([embed_of(cls, a) for a in entry["positive"]])
        if weighted:
            w = torch.tensor(entry["values"]).unsqueeze(1)
            proto = (w * e).sum(0) / len(entry["positive"])
        else:
            proto = e.mean(0)
        protos.append(proto)
    return torch.stack(protos)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", required=True, help="JSON from select_attributes")
    p.add_argument("--image-features", required=True, help="test <split>.pt")
    p.add_argument("--labels", required=True, help="test <split>_label.pt")
    p.add_argument("--classes", required=True, help="test <split>_classes.txt")
    p.add_argument("--protocol", choices=["attribute-only", "classname"],
                   default="attribute-only")
    p.add_argument("--weighted", action="store_true",
                   help="scale each attribute's vote by its selection value")
    p.add_argument("--pool", help="pool txt (attribute-only fast path)")
    p.add_argument("--attribute-embeddings", help=".pt (attribute-only fast path)")
    p.add_argument("--model", default="RN50", help="CLIP model for prompt encoding")
    args = p.parse_args()

    selection = json.load(open(args.selection))
    classes = [l.strip() for l in open(args.classes) if l.strip()]
    v = torch.load(args.image_features, map_location="cpu").float()
    y = torch.load(args.labels, map_location="cpu")

    if args.protocol == "attribute-only" and args.pool and args.attribute_embeddings:
        pool = load_pool(args.pool)
        embeds = torch.load(args.attribute_embeddings, map_location="cpu").float()
        index = {a: i for i, a in enumerate(pool)}
        embed_of = lambda cls, a: embeds[index[a]]
    else:
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load(args.model, device=device)
        prompts, seen = [], {}
        for cls in classes:
            for a in selection[cls]["positive"]:
                text = a if args.protocol == "attribute-only" else compose_prompt(cls, a)
                if text not in seen:
                    seen[text] = len(prompts)
                    prompts.append(text)
        all_embeds = encode_texts(model, prompts, device)
        embed_of = lambda cls, a: all_embeds[seen[
            a if args.protocol == "attribute-only" else compose_prompt(cls, a)]]

    protos = class_prototypes(selection, classes, embed_of, args.weighted)
    pred = (v @ protos.T).argmax(dim=1)
    acc = (pred == y).float().mean().item() * 100
    print(f"top-1 accuracy: {acc:.2f}%  "
          f"({args.protocol}, {'weighted' if args.weighted else 'unweighted'})")


if __name__ == "__main__":
    main()
