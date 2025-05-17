import numpy as np


def xyxy_to_cxcywh(xyxy):
    # 计算中心点坐标 cx, cy
    cx = (xyxy[..., 0] + xyxy[..., 2]) / 2.0
    cy = (xyxy[..., 1] + xyxy[..., 3]) / 2.0

    # 计算宽度和高度
    w = xyxy[..., 2] - xyxy[..., 0]
    h = xyxy[..., 3] - xyxy[..., 1]

    # 组合结果
    return np.stack((cx, cy, w, h), axis=-1)
