import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import albumentations as A

from gmdet.data import augment
from gmdet.data.base import BaseDataset
from gmdet.data.utils import PIN_MEMORY, img2label_paths, verify_image_label

from gmdet.utils import LOGGER


def collate_fn(batch):
    batch = list(zip(*batch))
    return tuple(batch)


class YOLODataset(BaseDataset):
    def __init__(self, img_path, imgsz, data, task, transforms):
        self.task = task
        self.data = data
        self._transforms = transforms
        self._resize = augment.LongestMaxSize(imgsz)
        # self._mosica = augment.Mosaic(self.load_anno, len(self.ids), imgsz, p=0.3)
        self._normalize = augment.Normalize()
        super(YOLODataset, self).__init__(img_path, imgsz)

    def __getitem__(self, idx):
        # box 归一化问题
        batch = self.get_image_and_label(idx)
        batch = self._resize(**batch)

        if self._transforms is not None:
            batch = self._transforms(**batch)
            # batch = self._mosica(**batch)

        return self._normalize(**batch)

    def cache_labels(self):
        labels = []
        nm, nf, ne = 0, 0, 0
        total = len(self.im_files)
        desc = f"Scanning ..."
        results = []
        for im_file, lb_file in zip(self.im_files, self.label_files):
            results.append(verify_image_label(im_file, lb_file, len(self.data["names"])))

        pbar = tqdm(results, desc=desc, total=total)

        for im_file, lb, segments, nm_f, nf_f, ne_f in pbar:
            nm += nm_f
            nf += nf_f
            ne += ne_f
            if im_file:
                labels.append({
                    "image": im_file,
                    "labels": lb[:, 0],  # n 1
                    'bboxes': lb[:, 1:]})  # n 4
                if len(segments) > 1:
                    labels[-1].update({'mask': segments})

                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds"
        pbar.close()
        return labels

    def get_labels(self):
        self.label_files = img2label_paths(self.im_files)
        labels = self.cache_labels()

        if not labels:
            LOGGER.warning(f"No images found, training may not work correctly.")

        # Check if the dataset is all boxes or all segments
        len_cls = (len(lb["labels"]) for lb in labels)

        # 没有标签，可能存在错误
        if len_cls == 0:
            LOGGER.warning(f"No labels found, training may not work correctly.")

        return labels


def build_yolo_dataset(img_path, imgsz, data, task, mode='train'):
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

    return YOLODataset(img_path,
                       imgsz=imgsz,
                       data=data,
                       task=task,
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
