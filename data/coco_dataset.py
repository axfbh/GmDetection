from typing import List
import os
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader

import albumentations as A

from data import augment
from utils.numpy_utils import xyxy_to_cxcywh, cxcywh_to_xyxy
import cv2
from ultralytics.data.augment import Albumentations


def collate_fn(batch):
    batch = list(zip(*batch))
    return tuple(batch)


PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


def visualize_bbox(img, bbox, w, h, color=(255, 0, 0), thickness=2):
    test_color = (255, 255, 255)  # White

    """Visualizes a single bounding box on the image"""
    for box in bbox:
        x_min, y_min, x_max, y_max = cxcywh_to_xyxy(box)

        x_min = int(x_min * w)
        y_min = int(y_min * h)
        x_max = int(x_max * w)
        y_max = int(y_max * h)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    return img


def coco_to_boxes(image, target):
    w, h = image.size

    anno = [obj for obj in target if 'iscrowd' not in obj or obj['iscrowd'] == 0]

    boxes = [obj["bbox"] for obj in anno]
    # guard against no boxes via resizing
    boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    # xmin, xmax
    boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=w)
    # ymin, ymax
    boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=h)

    classes = [obj["category_id"] for obj in anno]
    classes = np.array(classes, dtype=np.int64)

    # ymax > ymin and xmax > xmin
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = xyxy_to_cxcywh(boxes)
    boxes = boxes[keep] / np.array([w, h, w, h])
    classes = classes[keep]

    target = {"image": np.array(image), "bboxes": boxes, "labels": classes}

    return target


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, imgsz, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.imgsz = imgsz
        self._transforms = transforms
        self._resize = augment.LongestMaxSize(self.imgsz)
        self._normalize = augment.Normalize()
        self._mosica = augment.Mosaic(self.load_anno, len(self.ids), imgsz)

    def __getitem__(self, idx):
        # box 归一化问题
        batch = self.load_anno(idx)
        batch = self._resize(**batch)

        if self._transforms is not None:
            batch = self._transforms(**batch)
            batch = self._mosica(**batch)

        h, w, c = batch['image'].shape
        img = visualize_bbox(batch['image'], batch['bboxes'], w, h)
        from PIL import Image
        Image.fromarray(img).show()

        return self._normalize(**batch)

    def load_anno(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        return coco_to_boxes(img, target)


def build_coco_dataset(img_folder, ann_file, imgsz, mode='train', return_masks=False):
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

    return CocoDetection(img_folder,
                         ann_file,
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
