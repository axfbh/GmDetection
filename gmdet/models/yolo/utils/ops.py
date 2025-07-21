import cv2
from skimage import io

import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator as AG


def make_grid(h, w, sh=1, sw=1, device='cpu'):
    shifts_x = torch.arange(0, w, device=device) * sw
    shifts_y = torch.arange(0, h, device=device) * sh

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    shifts = torch.stack((shift_x, shift_y), dim=1)
    return shifts


class AnchorGenerator(AG):

    def forward(self, image_sizes: torch.Tensor, feature_maps: torch.Tensor):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        strides = [
            [
                image_sizes[0] // g[0],
                image_sizes[1] // g[1],
            ]
            for g in grid_sizes
        ]

        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        strides = [torch.tensor(st, device=device) for st in strides]
        return anchors_over_all_feature_maps, strides


def show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_FREERATIO)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)


def imread(path: str):
    img = cv2.cvtColor(io.imread(path), cv2.COLOR_RGB2BGR)
    return img

