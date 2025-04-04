from typing import Optional, List, Union, Dict

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from dataset.utils import NestedTensor

# import ops
# from models.backbone import cspdarknet
# from models.backbone import darknet
# from models.backbone import elandarknet
# from utils.torch_utils import NestedTensor

from yolo.models.backbone import cspdarknet


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, layers_to_train: List, return_interm_layers: Dict):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        self.body = IntermediateLayerGetter(backbone, return_layers=return_interm_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self,
                 name: str,
                 layers_to_train: List,
                 return_interm_layers: Dict,
                 norm_layer=nn.BatchNorm2d,
                 pretrained=True):
        backbone = getattr(cspdarknet, name)(pretrained=pretrained, norm_layer=norm_layer)
        super().__init__(backbone, layers_to_train, return_interm_layers)

#
# if __name__ == '__main__':
#     b = Backbone('cpsdarknetv4n',
#                  ['crossStagePartial3', 'crossStagePartial4'],
#                  {
#                      'crossStagePartial3': '0',
#                      'crossStagePartial4': '1',
#                  })
#     print(1)
