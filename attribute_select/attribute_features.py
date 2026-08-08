"""Compute attribute similarity features: s_j = tau * cos(v, t_j)  (paper Eq. 1).

Inputs are the L2-normalized image embeddings from extract_features and the
L2-normalized attribute embeddings from encode_attributes, so the cosine is a
plain matrix product. Output: float tensor [N, M].
"""
import argparse

import torch

from . import TEMPERATURE


def similarity_features(image_embeds, attribute_embeds, batch_size=100_000):
    chunks = []
    for i in range(0, image_embeds.shape[0], batch_size):
        chunks.append(TEMPERATURE * image_embeds[i:i + batch_size] @ attribute_embeds.T)
    return torch.cat(chunks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-features", required=True, help=".pt from extract_features")
    p.add_argument("--attribute-embeddings", required=True, help=".pt from encode_attributes")
    p.add_argument("--out", required=True, help="output .pt path")
    args = p.parse_args()

    v = torch.load(args.image_features, map_location="cpu").float()
    t = torch.load(args.attribute_embeddings, map_location="cpu").float()
    s = similarity_features(v, t)
    torch.save(s, args.out)
    print(f"saved similarity features {tuple(s.shape)} -> {args.out}")


if __name__ == "__main__":
    main()
