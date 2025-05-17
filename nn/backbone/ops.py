from typing import List, Union, Dict

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from data.ops import NestedTensor

from nn import backbone as nn_backbone


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, layers_to_train: List, return_interm_layers: Dict):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        self.body = IntermediateLayerGetter(backbone, return_layers=return_interm_layers)

    def forward(self, tensor_list: Union[NestedTensor, Tensor]):

        if isinstance(tensor_list, NestedTensor):
            xs = self.body(tensor_list.tensors)
            out: Dict[str, NestedTensor] = {}
            for name, x in xs.items():
                m = tensor_list.mask
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                out[name] = NestedTensor(x, mask)
            return out

        return self.body(tensor_list)


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self,
                 name: str,
                 layers_to_train: List,
                 return_interm_layers: Dict,
                 *args,
                 **kwargs):
        backbone = getattr(nn_backbone, name)(*args, **kwargs)
        super().__init__(backbone, layers_to_train, return_interm_layers)
