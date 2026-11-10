"""Training-only JPEG augmentation with unchanged spatial labels."""
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .degradations import degrade, to_tensor


class JpegTrainingDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        # 50% unchanged, 50% recompressed; each JPEG quality gets probability 1/6.
        condition = ("clean", "clean", "clean", "jpeg90", "jpeg70", "jpeg50")[torch.randint(6, ()).item()]
        if condition != "clean":
            image = Image.fromarray((sample["image"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
            sample = dict(sample, image=to_tensor(degrade(image, condition)))
        return sample
