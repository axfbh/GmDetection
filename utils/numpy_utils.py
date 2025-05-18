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


def cxcywh_to_xyxy(bboxes: np.ndarray) -> np.ndarray:
    """
    将 BBox 从 [cx, cy, w, h] 转换为 [x1, y1, x2, y2]
    :param bboxes: 输入 BBox 数组，形状为 (N, 4+)（支持附加参数如类别）
    :return: 转换后的 BBox 数组，形状为 (N, 4+)
    """
    # 提取 cx, cy, w, h
    cx = bboxes[..., 0]
    cy = bboxes[..., 1]
    w = bboxes[..., 2]
    h = bboxes[..., 3]

    # 计算角点坐标
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # 合并结果并保留附加参数（如类别）
    xyxy = np.concatenate([
        x1[..., np.newaxis],
        y1[..., np.newaxis],
        x2[..., np.newaxis],
        y2[..., np.newaxis],
        bboxes[..., 4:]  # 保留类别等额外字段
    ], axis=-1)

    return xyxy
