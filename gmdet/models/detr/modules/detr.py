from torch import nn
from torchvision.ops.misc import FrozenBatchNorm2d

from gmdet.nn.backbone import Backbone
from gmdet.nn.head.detr_head import DetrHead
from gmdet.nn.transformer import Transformer

from gmdet.models.detr.utils.position_encoding import PositionEmbeddingSine
from gmdet.models.detr.utils.detr_loss import DETRLoss
from gmdet.models.detr.utils.matcher import HungarianMatcher
from gmdet.utils import LOGGER


class Detr(nn.Module):
    def __init__(self, cfg, nc=None):
        super(Detr, self).__init__()
        self.yaml = cfg

        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value

        self.hidden_dim = self.yaml['hidden_dim']
        self.num_heads = self.yaml['num_heads']
        self.dim_feedforward = self.yaml['dim_feedforward']
        self.enc_layers = self.yaml['enc_layers']
        self.dec_layers = self.yaml['dec_layers']
        self.num_channels = self.yaml['num_channels']
        self.num_queries = self.yaml['num_queries']
        self.nc = self.yaml['nc']
        self.aux_loss = self.yaml['aux_loss']
        self.pre_norm = self.yaml['pre_norm']
        self.dropout = self.yaml['dropout']

        self.backbone = Backbone(name='resnet50',
                                 layers_to_train=['layer2', 'layer3', 'layer4'],
                                 return_interm_layers={'layer4': "0"},
                                 pretrained=True,
                                 norm_layer=FrozenBatchNorm2d)

        N_steps = self.hidden_dim // 2

        self.position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

        self.transformer = Transformer(
            d_model=self.hidden_dim,
            dropout=self.dropout,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            num_encoder_layers=self.enc_layers,
            num_decoder_layers=self.dec_layers,
            normalize_before=self.pre_norm,
            return_intermediate_dec=True,
        )

        self.head = DetrHead(self.hidden_dim, self.hidden_dim, 4, 3, nc + 1, aux_loss=self.aux_loss)
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        self.input_proj = nn.Conv2d(self.num_channels, self.hidden_dim, kernel_size=1)

    def forward(self, batch):
        samples = batch[0]

        features = self.backbone(samples)['0']

        pos = self.position_embedding(features)

        src, mask = features.decompose()
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)[0]

        # ----------- train -----------
        if self.training:
            targets = batch[2]
            preds = self.head(hs)
            return self.loss(preds, targets)

        return self.head(hs, self.args.imgsz)[0]

    def loss(self, preds, targets):
        if getattr(self, "criterion", None) is None:
            matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)

            self.criterion = DETRLoss(self.num_classes,
                                      matcher=matcher,
                                      eos_coef=0.1)

        return self.criterion(preds, targets)
