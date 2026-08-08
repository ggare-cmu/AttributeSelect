"""Extract L2-normalized CLIP image embeddings for a dataset.

The dataset is an ImageFolder-style directory (one subdirectory per class).
Outputs, under --out-dir:
  <split>.pt           float tensor [N, D], L2-normalized image embeddings
  <split>_label.pt     long tensor [N], class indices (ImageFolder order)
  <split>_classes.txt  one class name per line, index order
"""
import argparse
import pathlib

import clip
import torch
from torchvision import datasets


def extract(model, dataset, device, batch_size=64, num_workers=8):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    features, labels = [], []
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            f = model.encode_image(images.to(device)).float().cpu()
            features.append(f.clone())
            labels.append(targets.clone())
    features = torch.cat(features)
    features = features / features.norm(dim=-1, keepdim=True)
    return features, torch.cat(labels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True,
                   help="ImageFolder root (one subdirectory per class)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="train", help="output file prefix")
    p.add_argument("--model", default="RN50", help="CLIP model name")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model, device=device)
    dataset = datasets.ImageFolder(args.images, transform=preprocess)

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    features, labels = extract(model, dataset, device,
                               args.batch_size, args.num_workers)
    torch.save(features, out / f"{args.split}.pt")
    torch.save(labels, out / f"{args.split}_label.pt")
    (out / f"{args.split}_classes.txt").write_text(
        "\n".join(dataset.classes) + "\n")
    print(f"saved {tuple(features.shape)} embeddings to {out}/{args.split}.pt")


if __name__ == "__main__":
    main()
