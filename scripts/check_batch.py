"""Check deterministic samples, Windows workers, and image/mask batch alignment."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from doctrace.dataset import DocTamperDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--database", default="data/raw/DocTamperV1-TrainingSet")
    parser.add_argument("--output", default="outputs/batch_preview.png")
    args = parser.parse_args()
    dataset = DocTamperDataset(args.database)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(dataset), generator=generator)[:8].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=4,
                        num_workers=args.workers, shuffle=False)
    collected = []
    for batch in loader:
        images, masks = batch["image"], batch["mask"]
        assert images.shape == (4, 3, 512, 512), images.shape
        assert masks.shape == (4, 1, 512, 512), masks.shape
        assert images.dtype == masks.dtype == torch.float32
        assert torch.isfinite(images).all() and torch.isfinite(masks).all()
        assert ((images >= 0) & (images <= 1)).all()
        assert ((masks == 0) | (masks == 1)).all()
        print(f"Indices: {batch['index'].tolist()} | images: {tuple(images.shape)} | masks: {tuple(masks.shape)}")
        for image, mask, index in zip(images, masks, batch["index"]):
            collected.append((image.permute(1, 2, 0).numpy(), mask[0].numpy(), int(index)))
    dataset.close()
    fig, axes = plt.subplots(len(collected), 3, figsize=(12, 3 * len(collected)))
    for row, (image, mask, index) in enumerate(collected):
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask > 0] = [1, 0, 0, 0.45]
        axes[row, 0].imshow(image)
        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(image)
        axes[row, 2].imshow(overlay)
        axes[row, 0].set_title(f"Image {index}")
        axes[row, 1].set_title(f"Mask: {mask.mean():.2%} positive")
        axes[row, 2].set_title("Ground-truth overlay")
        for ax in axes[row]:
            ax.axis("off")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=100)
    plt.close(fig)
    print(f"PASS: {len(collected)} samples, workers={args.workers}")
    print(f"Preview: {output.resolve()}")


if __name__ == "__main__":
    main()
