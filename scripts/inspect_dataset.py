from io import BytesIO
from pathlib import Path

import lmdb
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATABASE = Path("data/raw/DocTamperV1-TrainingSet")
OUTPUT = Path("outputs/training_preview.png")


def main():
    env = lmdb.open(
        str(DATABASE),
        readonly=True,
        lock=False,
        create=False,
        readahead=False,
    )

    try:
        with env.begin() as txn:
            # Inspect key families without printing binary values.
            prefixes = {}
            for key, _ in txn.cursor():
                prefix = key.decode("utf-8", errors="replace").rsplit("-", 1)[0]
                prefixes[prefix] = prefixes.get(prefix, 0) + 1

            print("Key families:", prefixes)

            image_bytes = txn.get(b"image-000000000")
            mask_bytes = txn.get(b"label-000000000")

        if image_bytes is None or mask_bytes is None:
            print("Expected image/label pair not found.")
            print("Send the key families above so we can adjust the loader.")
            return

        image = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
        mask = np.array(Image.open(BytesIO(mask_bytes)))

        print("Image shape:", image.shape)
        print("Mask shape:", mask.shape)
        print("Mask values:", np.unique(mask))

        if mask.ndim != 2 or mask.shape != image.shape[:2]:
            print("Unexpected mask dimensions; inspect before proceeding.")
            return

        # Nonzero pixels are displayed as the candidate tampering region.
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask > 0] = [1.0, 0.0, 0.0, 0.45]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title("Document")
        axes[1].imshow(mask, cmap="gray", interpolation="nearest")
        axes[1].set_title("Raw label mask")
        axes[2].imshow(image)
        axes[2].imshow(overlay)
        axes[2].set_title("Nonzero label overlay")

        for ax in axes:
            ax.axis("off")

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(OUTPUT, dpi=160)
        plt.close(fig)
        print("Preview saved to:", OUTPUT.resolve())
    finally:
        env.close()


if __name__ == "__main__":
    main()