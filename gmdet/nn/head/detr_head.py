import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.misc import MLP
from torchvision.ops.boxes import box_convert


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class DetrHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_classes, aux_loss=False):
        super(DetrHead, self).__init__()
        self.nc = num_classes
        self.aux_loss = aux_loss
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        hidden_channels = [k for k in h + [output_dim]]

        self.bbox_embed = MLP(input_dim, hidden_channels)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

    def forward(self, x, imgsz=None):
        outputs_class = self.class_embed(x)
        outputs_coord = self.bbox_embed(x).sigmoid()
        outputs = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        if self.aux_loss:
            outputs['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.training:
            return outputs

        z = self._inference(outputs, imgsz)
        return z, outputs

    def _inference(self, outputs, imgsz):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = box_convert(out_bbox, 'cxcywh', 'xyxy')

        boxes = boxes * imgsz

        return torch.cat([boxes, scores[..., None], labels[..., None]], -1)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
