import random
from typing import Literal, Sequence, Union, Dict, Any, List, overload
from itertools import chain

import numpy as np

import cv2
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2


class LongestMaxSize:
    def __init__(self,
                 max_size: Union[int, Sequence[int]] = 1024,
                 interpolation: int = cv2.INTER_LINEAR,
                 format: Literal["coco", "pascal_voc", "albumentations", "yolo"] = "yolo",
                 always_apply: bool = False,
                 p: float = 1):
        T = [
            A.LongestMaxSize(max_size=max_size, interpolation=interpolation, always_apply=always_apply, p=p)
        ]

        self.fit_transform = A.Compose(T, A.BboxParams(format=format, label_fields=['labels']))

    def __call__(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)


class Normalize:
    def __init__(self,
                 mean: None = (0.485, 0.456, 0.406),
                 std: None = (0.229, 0.224, 0.225),
                 max_pixel_value: Union[float, None] = 255.0,
                 normalization: Literal[
                     "standard", "image", "image_per_channel", "min_max", "min_max_per_channel"] = "standard",
                 always_apply: bool = False,
                 p: float = 1.0):
        T = [
            A.Normalize(mean, std, max_pixel_value, normalization, always_apply, p),
            ToTensorV2()
        ]
        self.fit_transform = A.Compose(T)

    def __call__(self, *args, **kwargs):
        image = self.fit_transform(image=kwargs['image'])['image']
        target = {"boxes": torch.as_tensor(kwargs['bboxes']),
                  "labels": torch.as_tensor(kwargs['labels'], dtype=torch.long)}
        return image, target


class Mosaic:
    def __init__(
            self,
            read_anno,  # 其他三张样本（包含 image/bboxes/masks）
            size,
            output_size=640,
            format: Literal["coco", "pascal_voc", "albumentations", "yolo"] = "yolo",
            always_apply=False,
            p=0.5):

        assert output_size % 32 == 0, "Mosaic output_size 必须能被 32 整除"

        self.read_anno = read_anno
        self.size = size

        self.p = p
        self.always_apply = always_apply

        self.output_size = output_size
        self.output_size_half = output_size // 2

        T = [
            A.SmallestMaxSize(max_size=output_size),
            A.RandomCrop(self.output_size_half, self.output_size_half)
        ]

        self.resize = A.Compose(T, A.BboxParams(format=format, label_fields=['labels'], min_visibility=0.1))

    def __call__(self, *args, **kwargs):
        if self.always_apply or random.random() < self.p:
            batch = self.resize(*args, **kwargs)
            batches = [batch] + self.get_cache_batch()
            image = self.apply([b["image"] for b in batches])
            bboxes = self.apply_to_bboxes([b["bboxes"] for b in batches])
            batch["image"] = image
            batch["bboxes"] = bboxes
            batch["labels"] = list(chain(*[b["labels"] for b in batches]))
            return batch
        return kwargs

    def apply(self, images):
        mosaic_img = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)

        # 定义四个子图位置（左上、右上、左下、右下）
        positions = [
            (0, 0),  # 左上
            (self.output_size_half, 0),  # 左上
            (0, self.output_size_half),  # 左下
            (self.output_size_half, self.output_size_half)  # 右下
        ]

        for idx, (x, y) in enumerate(positions):
            mosaic_img[y:y + self.output_size_half, x:x + self.output_size_half] = images[idx]

        return mosaic_img

    def apply_to_bboxes(self, bboxes):
        # 合并四张子图的 BBox，并根据位置调整坐标
        mosaic_bboxes = []

        offsets = [
            (0, 0),  # 索引 0: 左上
            (self.output_size_half, 0),  # 索引 1: 右上
            (0, self.output_size_half),  # 索引 2: 左下
            (self.output_size_half, self.output_size_half),  # 索引 3: 右下
        ]

        for (dx, dy), bbox in zip(offsets, bboxes):
            for cx, cy, w, h in bbox:
                # 解包 BBox 坐标: [x_min, y_min, x_max, y_max, ...(其他参数)]
                # 确保坐标不越界
                cx = (np.clip(cx * self.output_size_half, 0, self.output_size_half) + dx) / self.output_size
                cy = (np.clip(cy * self.output_size_half, 0, self.output_size_half) + dy) / self.output_size
                w = w * self.output_size_half / self.output_size
                h = h * self.output_size_half / self.output_size

                # 将调整后的 BBox 添加到列表
                mosaic_bboxes.append(np.array([cx, cy, w, h]))

        return np.array(mosaic_bboxes)

    def get_cache_batch(self) -> List[Dict[str, Any]]:
        ids = np.random.randint(0, self.size, 3, dtype=np.int64)

        return [self.resize(**self.read_anno(int(i))) for i in ids]
