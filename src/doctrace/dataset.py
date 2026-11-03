"""Read-only DocTamper LMDB dataset with process-local connections."""
from io import BytesIO
from pathlib import Path
import os

import lmdb
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class DocTamperDataset(Dataset):
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        self._env = None
        self._pid = None
        env = self._open()
        try:
            with env.begin() as txn:
                count = txn.get(b"num-samples")
                if count is None:
                    raise ValueError("Missing num-samples key")
                self.length = int(count)
        finally:
            env.close()

    def _open(self):
        return lmdb.open(self.path, readonly=True, lock=False, create=False,
                         readahead=False, meminit=False)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        if self._env is None or self._pid != os.getpid():
            self.close()
            self._env = self._open()
            self._pid = os.getpid()
        with self._env.begin() as txn:
            image_bytes = txn.get(f"image-{index:09d}".encode())
            mask_bytes = txn.get(f"label-{index:09d}".encode())
        if image_bytes is None or mask_bytes is None:
            raise KeyError(f"Missing image/label pair at index {index}")
        with Image.open(BytesIO(image_bytes)) as image:
            rgb = np.array(image.convert("RGB"), copy=True)
        with Image.open(BytesIO(mask_bytes)) as image:
            mask = np.array(image, copy=True)
        if mask.ndim != 2 or mask.shape != rgb.shape[:2]:
            raise ValueError(f"Invalid image/mask shapes at index {index}")
        if not np.isin(mask, [0, 1, 255]).all():
            raise ValueError(f"Unexpected mask values at index {index}")
        # Preserve resolution and orientation; no augmentation or normalization yet.
        image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)
        mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        return {"image": image_tensor, "mask": mask_tensor, "index": index}

    def close(self):
        if self._env is not None:
            self._env.close()
        self._env = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_pid"] = None
        return state
