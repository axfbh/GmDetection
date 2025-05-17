import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from typing import Literal, Sequence, Union


class LongestMaxSize:
    def __init__(self,
                 max_size: Union[int, Sequence[int]] = 1024,
                 interpolation: int = cv2.INTER_LINEAR,
                 always_apply: Union[bool, None] = None,
                 p: float = 1,
                 format: Literal["coco", "pascal_voc", "albumentations", "yolo"] = "yolo"):
        T = [
            A.LongestMaxSize(max_size=max_size, interpolation=interpolation, always_apply=always_apply, p=p)
        ]

        self.fit_transform = A.Compose(T, A.BboxParams(format=format, label_fields=['labels']))

    def __call__(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)


# A.Normalize


class Normalize:
    def __init__(self,
                 mean: None = (0.485, 0.456, 0.406),
                 std: None = (0.229, 0.224, 0.225),
                 max_pixel_value: Union[float, None] = 255.0,
                 normalization: Literal[
                     "standard", "image", "image_per_channel", "min_max", "min_max_per_channel"] = "standard",
                 always_apply: Union[bool, None] = None,
                 p: float = 1.0):
        T = [
            A.Normalize(mean, std, max_pixel_value, normalization, always_apply, p),
            ToTensorV2()
        ]
        self.fit_transform = A.Compose(T)

    def __call__(self, *args, **kwargs):
        image = self.fit_transform(image=kwargs['image'])['image']
        target = {"boxes": torch.from_numpy(kwargs['bboxes']),
                  "labels": torch.as_tensor(kwargs['labels'], dtype=torch.long)}
        return image, target
