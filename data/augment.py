from typing import Literal, Sequence, Union, Dict, Any, List, overload
import numpy as np

import cv2
import torch

import albumentations as A
from albumentations import DualTransform
from albumentations.pytorch import ToTensorV2


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

# class Mosaic4(DualTransform):
#     def __init__(
#             self,
#             read_anno,  # 其他三张样本（包含 image/bboxes/masks）
#             size,
#             output_size=640,
#             always_apply=False,
#             p=0.5,
#             format: Literal["coco", "pascal_voc", "albumentations", "yolo"] = "yolo"):
#         super().__init__(p, always_apply)
#         self.size = size
#         self.read_anno = read_anno
#         self.output_size = output_size
#
#         T = [
#             A.SmallestMaxSize(max_size=output_size),
#             A.RandomCrop(output_size, output_size)
#         ]
#
#         self.resize = A.Compose(T, A.BboxParams(format=format, label_fields=['labels']))
#
#     def get_params(self) -> Dict[str, List[Dict[str, Any]]]:
#         ids = np.random.randint(0, self.size, 3, dtype=np.int64)
#
#         batches = [self.resize(**self.read_anno(int(i))) for i in ids]
#
#         return {"batches": batches}
#
#     def apply(self, image, batches, **params):
#         # 初始化输出图像
#         mosaic_img = np.zeros((self.output_size * 2, self.output_size * 2, 3), dtype=np.uint8)
#
#         # 定义四个子图位置（左上、右上、左下、右下）
#         positions = [
#             (0, 0),  # 左上
#             (0, self.output_size),  # 右上
#             (self.output_size, 0),  # 左下
#             (self.output_size, self.output_size)  # 右下
#         ]
#         images = [image] + [b['image'] for b in batches]
#
#         for idx, (x, y) in enumerate(positions):
#             mosaic_img[x:x + self.output_size, y:y + self.output_size] = images[idx]
#
#         return mosaic_img
#
#     def apply_to_bboxes(self, bboxes, batches, **params):
#         # 合并四张子图的 BBox，并根据位置调整坐标
#         mosaic_bboxes = []
#         batches = [bboxes] + [b['bboxes'] for b in batches]
#
#         positions = [
#             (0, 0),  # 左上
#             (0, self.output_size),  # 右上
#             (self.output_size, 0),  # 左下
#             (self.output_size, self.output_size)  # 右下
#         ]
#
#         for (x, y), meta in zip(positions, batches):
#             for bbox in meta["bboxes"]:
#                 # 解包 BBox 坐标: [x_min, y_min, x_max, y_max, ...(其他参数)]
#                 # x_min, y_min, x_max, y_max = bbox[:4]
#
#                 # 确保坐标不越界
#                 # x_min = np.clip(x_min + x, 0, self.output_size)
#                 # y_min = np.clip(y_min + y, 0, self.output_size)
#                 # x_max = np.clip(x_max + x, 0, self.output_size)
#                 # y_max = np.clip(y_max + y, 0, self.output_size)
#
#                 # 将调整后的 BBox 添加到列表
#                 mosaic_bboxes.append(np.array([x_min, y_min, x_max, y_max]))
#
#         return mosaic_bboxes
#
# def apply_to_masks(self, masks, **params):
#     # 合并四张子图的 Mask，并根据位置调整坐标
#     mosaic_masks = []
#     for meta in self.sub_img_meta:
#         for mask in meta["masks"]:
#             # 调整 Mask 尺寸
#             mask_resized = A.resize(
#                 mask,
#                 height=self.output_size // 2,
#                 width=self.output_size // 2
#             )
#
#             # 创建全零矩阵（马赛克尺寸）
#             full_mask = np.zeros((self.output_size, self.output_size), dtype=np.uint8)
#             full_mask[
#             meta["offset_y"]: meta["offset_y"] + self.output_size // 2,
#             meta["offset_x"]: meta["offset_x"] + self.output_size // 2
#             ] = mask_resized
#
#             mosaic_masks.append(full_mask)
#
#     return mosaic_masks
#
# @property
# def targets(self):
#     return {"image": self.apply, "bboxes": self.apply_to_bboxes, "masks": self.apply_to_masks}

#
# class Mosaic:
#     def __init__(
#             self,
#             read_anno,  # 其他三张样本（包含 image/bboxes/masks）
#             size,
#             output_size=640,
#             always_apply=False,
#             p=0.5,
#             format: Literal["coco", "pascal_voc", "albumentations", "yolo"] = "yolo"):
#         T = [
#             A.SmallestMaxSize(max_size=output_size),
#             A.RandomCrop(output_size, output_size)
#         ]
#
#         self.resize = A.Compose(T, A.BboxParams(format=format, label_fields=['labels']))
#
#         T = [
#             Mosaic4(read_anno, size, output_size, always_apply, p, format),
#             A.RandomCrop(output_size, output_size)
#         ]
#
#         self.fit_transform = A.Compose(T, A.BboxParams(format=format, label_fields=['labels']))
#
#     def __call__(self, *args, **kwargs):
#         batch = self.resize(*args, **kwargs)
#         return self.fit_transform(**batch)
