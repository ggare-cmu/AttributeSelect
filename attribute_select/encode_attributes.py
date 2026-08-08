"""Encode an attribute pool with CLIP's text encoder.

Each attribute string is encoded BARE: no prompt template and no class name.
This is deliberate (paper Sec. 4): a shared template or class name would
correlate all of a class's attribute scores through the shared words.

Input:  a text file with one attribute per line.
Output: float tensor [M, D], L2-normalized attribute text embeddings.
"""
import argparse

import clip
import torch


def encode_texts(model, texts, device, batch_size=256):
    tokens = torch.cat([clip.tokenize(t, truncate=True) for t in texts])
    chunks = []
    with torch.inference_mode():
        for i in range(0, len(tokens), batch_size):
            f = model.encode_text(tokens[i:i + batch_size].to(device))
            chunks.append(f.float().cpu())
    embeds = torch.cat(chunks)
    return embeds / embeds.norm(dim=-1, keepdim=True)


def load_pool(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", required=True, help="txt file, one attribute per line")
    p.add_argument("--out", required=True, help="output .pt path")
    p.add_argument("--model", default="RN50", help="CLIP model name")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(args.model, device=device)
    pool = load_pool(args.pool)
    embeds = encode_texts(model, pool, device)
    torch.save(embeds, args.out)
    print(f"encoded {len(pool)} attributes -> {args.out} {tuple(embeds.shape)}")


if __name__ == "__main__":
    main()
