import os

import torch
from torch.utils.data import DataLoader

import albumentations as A

from data import augment
from data.base import BaseDataset
from data.utils import PIN_MEMORY


def collate_fn(batch):
    batch = list(zip(*batch))
    return tuple(batch)


class DetectDataset(BaseDataset):
    def __init__(self, img_path, imgsz, transforms):
        super(DetectDataset, self).__init__(img_path, imgsz)

        self._transforms = transforms
        self._resize = augment.LongestMaxSize(self.imgsz)
        self._normalize = augment.Normalize()
        # self._mosica = augment.Mosaic(self.load_anno, len(self.ids), imgsz, p=0.3)

    def __getitem__(self, idx):
        # box 归一化问题
        batch = self.get_image_and_label(idx)
        batch = self._resize(**batch)

        if self._transforms is not None:
            batch = self._transforms(**batch)
            # batch = self._mosica(**batch)

        return self._normalize(**batch)


def build_detect_dataset(img_path, imgsz, mode='train'):
    T = [
        A.Blur(p=0.01),
        A.MedianBlur(p=0.01),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
        A.RandomBrightnessContrast(p=0.0),
        A.RandomGamma(p=0.0),
        A.ImageCompression(quality_lower=75, p=0.0),
    ]
    transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]))

    return DetectDataset(img_path,
                         imgsz=imgsz,
                         transforms=transform if mode == 'train' else None)


def build_dataloader(dataset,
                     batch,
                     workers=3,
                     shuffle=False,
                     persistent_workers=False):
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers

    return DataLoader(dataset=dataset,
                      batch_size=batch,
                      shuffle=shuffle,
                      num_workers=nw,
                      pin_memory=PIN_MEMORY,
                      collate_fn=collate_fn,
                      drop_last=True,
                      persistent_workers=persistent_workers)
