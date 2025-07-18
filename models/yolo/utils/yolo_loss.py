from bisect import bisect_left

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.boxes import box_convert

from models.yolo.utils.ops import make_grid
from models.yolo.utils.iou_loss import all_iou_loss
from models.yolo.utils.boxes import dist2bbox, bbox2dist
from models.yolo.utils.tal import TaskAlignedAssigner, TaskNearestAssigner

torch.set_printoptions(precision=4, sci_mode=False)


class YoloAnchorBasedLoss(nn.Module):
    def __init__(self, model, topk=3):
        super(YoloAnchorBasedLoss, self).__init__()

        hyp = model.args
        m = model.head

        self.device = model.device
        self.hyp = hyp
        # yolo 小grid大anchor，大grid小anchor
        self.anchors = m.anchors
        self.nl = m.nl
        self.na = m.na
        self.nc = m.nc
        self.no = m.no
        ids = bisect_left([1, 3, 5], topk)
        self.alpha = [1, 2, 3][ids]
        self.gamma = [0, 0.5, 1][ids]

        self.hyp["box"] *= 3 / self.nl
        self.hyp["obj"] *= self.nc / 80 * 3 / self.nl
        self.hyp["cls"] *= (self.hyp.imgsz / 640) ** 2 * 3 / self.nl

        self.assigner = TaskNearestAssigner(anchor_t=self.hyp['anchor_t'], topk=topk, num_classes=self.nc)

        self.balance = [4.0, 1.0, 0.4]

        # Define criteria
        self.BCEcls = nn.BCEWithLogitsLoss()
        self.BCEobj = nn.BCEWithLogitsLoss()


class YoloAnchorFreeLoss(nn.Module):
    def __init__(self, model, tal_topk=10):
        super(YoloAnchorFreeLoss, self).__init__()

        hyp = model.args
        m = model.head

        # Define criteria
        self.device = model.device
        self.hyp = hyp
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.make_anchors = m.make_anchors
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(self.device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=self.device)


class YoloLossV4To7(YoloAnchorBasedLoss):

    def preprocess(self, targets, batch_size):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        counts = max([len(t['boxes']) for t in targets])
        out = torch.zeros(batch_size, counts, 5, device=self.device)
        if counts == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            for i, t in enumerate(targets):
                boxes = t['boxes']
                cls = t['labels']
                n = len(boxes)
                if n:
                    # cls - 1, 剔除背景标签影响
                    out[i, :n, 0] = cls
                    out[i, :n, 1:] = boxes

        return out

    def forward(self, preds, targets):
        loss = torch.zeros(3, dtype=torch.float32, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        preds = [
            xi.view(feats[0].shape[0], self.na, -1, self.no).split((4, 1, self.nc), -1) for xi in feats
        ]

        batch_size = feats[0].shape[0]
        imgsz = self.hyp.imgsz

        targets = self.preprocess(targets, batch_size)
        gt_cls, gt_cxys, gt_whs = targets.split((1, 2, 2), 2)  # cls, xyxy
        # 非填充的目标mask
        mask_gt = gt_cxys.sum(2, keepdim=True).gt_(0)  # [b,n_box,1]

        for i in range(self.nl):
            _, _, ng, _, _ = feats[i].shape

            stride = imgsz / ng

            target_bboxes, target_scores, anc_wh, fg_mask = self.assigner(
                self.anchors[i] / stride,
                make_grid(ng, ng, device=self.device),
                gt_cls,
                gt_cxys * ng,
                gt_whs * ng,
                mask_gt
            )

            pred_bboxes, pred_obj, pred_scores = preds[i]

            target_obj = torch.zeros_like(pred_obj)

            if fg_mask.any() > 0:
                pxy = pred_bboxes[..., :2].sigmoid() * self.alpha - self.gamma
                pwh = (pred_bboxes[..., 2:].sigmoid() * 2) ** 2 * anc_wh
                pred_bboxes = torch.cat([pxy, pwh], -1)
                iou = all_iou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], in_fmt='cxcywh', CIoU=True)

                loss[0] += (1.0 - iou).mean()

                iou = iou.detach().clamp(0).type(target_obj.dtype)
                target_obj[fg_mask] = iou[:, None]  # iou ratio

                if self.nc > 1:
                    loss[2] += self.BCEcls(pred_scores[fg_mask], target_scores[fg_mask])

            obji = self.BCEobj(pred_obj, target_obj)
            loss[1] += obji * self.balance[i]  # obj loss

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.obj
        loss[2] *= self.hyp.cls

        return loss.sum() * batch_size, {k: v for k, v in zip(['box_loss',
                                                               'obj_loss',
                                                               'cls_loss'], loss.detach())}  # loss(box, obj, cls)


class YoloLossV8(YoloAnchorFreeLoss):

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        # 不使用 dfl ，就是回归 4 个值
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            # pred_dist.dot([0,1,...,15])
            # pred_dist[0] * 0 + ... + pred_dist[15] * 15
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        counts = max([len(t['boxes']) for t in targets])
        out = torch.zeros(batch_size, counts, 5, device=self.device)
        if counts == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            for i, t in enumerate(targets):
                boxes = t['boxes']
                cls = t['labels']
                n = len(boxes)
                if n:
                    out[i, :n, 0] = cls
                    out[i, :n, 1:] = box_convert(boxes, 'cxcywh', 'xyxy').mul_(scale_tensor)

        return out

    def forward(self, preds, targets):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = self.hyp.imgsz

        # anchor_points：候选框中心点
        # stride_tensor：缩放尺度
        anchor_points, stride_tensor = self.make_anchors([imgsz, imgsz], preds)

        targets = self.preprocess(targets, batch_size, imgsz)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        # 非填充 bbox 样本索引 mask
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)  # [b,n_box,1]

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, {k: v for k, v in zip(['box_loss',
                                                               'cls_loss',
                                                               'dfl_loss'], loss.detach())}  # loss(box, obj, cls)


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
                F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
                + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """
        Criterion class for computing training losses during training.
    """

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = all_iou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            # reg_max 用于限制 target_ltrb 的长度到 reg_max
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl
