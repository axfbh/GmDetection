# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
from yolo.ops.iou_loss import all_iou_loss


class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=20, alpha=1.0, beta=6.0, eps=1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for horizontal bounding boxes."""
        return all_iou_loss(gt_bboxes, pd_bboxes, CIoU=True).clamp_(0)

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment. Reference code is available at
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]

        if self.n_max_boxes == 0:
            # device = gt_bboxes.device
            # return (
            #     torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
            #     torch.zeros_like(pd_bboxes).to(device),
            #     torch.zeros_like(pd_scores).to(device),
            #     torch.zeros_like(pd_scores[..., 0]).to(device),
            #     torch.zeros_like(pd_scores[..., 0]).to(device),
            # )
            return (
                None,
                None,
                torch.zeros(1, dtype=torch.bool, device=gt_labels.device),
                torch.zeros(1, dtype=torch.bool, device=gt_labels.device),
                None,
            )

        # 获取真实目标的mask（重叠），通过分类分数和预测框iou，寻找 topk 个正样本
        # mask_pos: 样本（重叠）mask
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        # 获取真实目标的mask、id（非重叠），通过iou，剔除网格重叠样本
        # mask_pos: 样本（非重叠）mask (b, n_max_box, h*w)
        # fg_mask: 样本（非重叠）mask (b, h*w)
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # 制作标签
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        # align_metric 根据 mask 剔除, 未被选中为 k 个正样本的样本
        # align_metric (b,n_max_boxes,h*w)
        align_metric *= mask_pos
        # pos_align_metrics, 寻找k个样本中，预测框分数最好的
        # pos_align_metrics: (b,n_max_boxes)
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        # pos_overlaps: (overlaps * mask_pos) 根据 mask 剔除 未被选中为 k 个正样本的样本,
        # amaxa: 寻找k个样本中，iou分数最好的
        # pos_overlaps: (b,n_max_boxes)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        # 归一次公式
        # overlaps : iou 最好的时候为 1
        # pos_align_metrics : 为 align_metric 最好的分数
        # 当 align_metric == pos_align_metrics，且 iou 为 1，那么 norm_align_metric = 1，即预测框已经预测的很好了
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """
            Get in_gts mask, (b, max_num_obj, h*w).
        """
        # 找出满足条件一：正样本的mask
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        # 计算每个正样本的分数，根据正样本的类别分数和iou
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # 找出满足条件二：正样本的mask
        mask_topk = self.select_topk_candidates(align_metric)
        # mask_topk：满足 topk 的样本 mask
        # mask_in_gts：满足在 gt_bboxes 内的样本 mask
        # mask_gt：非填充样本 mask
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        Select the positive anchor center in gt.

        Args:
            xy_centers (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, n_boxes, 4)

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
        """
        # 候选样本数量
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        # bbox 的 x1y1 x2y2
        # gt_bboxes.view(-1,1,4): (b * n_boxes,1,4) 所有 bboxes 汇聚在第一个维度
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        # 计算 候选框 中心点与 gt bboxes 的距离
        # xy_centers[None]: (1, n_anchors, 2)
        # xy_centers[None] - lt : (b * n_boxes, n_anchors, 2)
        # bbox_deltas.view(b, n_boxes, n_anchors, 4) ：每个 bboxex 有 n_anchors 个候选样本
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        # 每个 anchor 代表一个候选样本
        # 保留落在 gt bboxes 内的候选样本的 mask
        return bbox_deltas.amin(3).gt_(eps)  # shape(b, n_boxes, h*w)

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        Compute alignment metric given predicted and ground truth bounding boxes.
        """
        # 网格数量
        na = pd_bboxes.shape[-2]
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, n_max_boxes
        # ind[0]: (b, n_max_boxes) 每个图像有 n_max_boxes 个 bbox
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        # b, n_max_boxes
        ind[1] = gt_labels.squeeze(-1)
        # 1. pd_scores[ind[0], :, ind[1]] : 保留第二维度
        # 2. ind[0] 扩展第一维度成(b, n_max_boxes)
        # 3. ind[1] 的值，遍历 pd_scores 第三维度，ind[1].shape == ind[0].shape
        # 类别的分数，作为预测框的分数之一
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # pd_boxes.unsqueeze(1): (b, 1, n_anchors, 4)
        # pd_boxes.expand(-1, n_max_boxes, -1, -1): (b, n_max_boxes, na, 4) 膨胀每个网格n_max_boxes个目标
        # pd_boxes[mask_gt]: 取出对应位置值
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        # gt_boxes.unsqueeze(2): (b, n_max_boxes, 1, 4)
        # gt_boxes.expand(-1, -1, na, -1): (b, n_max_boxes, na, 4) 膨胀每个目标到na个网格中
        # gt_boxes[mask_gt]: 取出对应位置值
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        # 将计算好的 iou 存入对应位置
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        # 预测框的分数，根据类别的分数和框iou分数
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True):
        # 每个bbox选取k个网格作为正样本
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)

        # 获取每个bbox的k个网格的mask
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=metrics.device)
        count_tensor.scatter_(-1, topk_idxs, 1)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):

        # batch idx: (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        # batch_ind * self.n_max_boxes: [0, 1*n, 2*n, ..., b*n]
        # target_gt_idx + (batch_ind * self.n_max_boxes):
        # 图1[目标1的id设置在0, ..., 目标n的id设置在n-1]
        # 图2[目标1的id设置在n, ..., 目标n的id设置在2n]
        # target_gt_idx (b, h*w)
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        # gt_labels: (b, n, 1) -> (b*n)
        # target_labels: (b,h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # target_bboxes: (b, n, 2) -> (b*n, 2)
        # target_bboxes: (b, h*w, 2)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        #  target_scores: (b, h*w, c), One-Hot
        target_scores = torch.zeros((self.bs, target_labels.shape[1], self.num_classes),
                                    dtype=torch.int64,
                                    device=target_labels.device)  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        # 根据mask剔除没有样本的target_scores
        target_scores[~fg_mask.bool()] = 0
        # fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        # target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        If an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
        """
        # mask_pos: (b, n_max_boxes, h*w) -> (b, h*w)
        # 将每个网格的bbox合并一起，统计一个网格的bbox数量
        fg_mask = mask_pos.sum(-2)
        # 至少有一个重叠目标
        if fg_mask.max() > 1:
            # fg_mask: (b, 1, h*w) -> (b, n_max_boxes, h*w)
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)

            # 选取网格中iou最大的目标
            max_overlaps_idx = overlaps.argmax(-2)  # (b, h*w)

            # non_overlaps: (b, n_max_boxes, h*w)
            non_overlaps = torch.ones(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)

            # 重叠网格全部置为 0
            non_overlaps[mask_multi_gts] = 0

            # 重叠网格最大分数置置为 1
            non_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            # 真实样本的mask = 真实样本（非重叠）* 非填充样本 -> (b, n_max_boxes, h*w)
            mask_pos = non_overlaps * mask_pos

            # 真实样本的mask (b, h*w)
            fg_mask = mask_pos.sum(-2)

        # 真实样本的id (b, h*w)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


class TaskNearestAssigner(nn.Module):
    def __init__(self, topk=3, num_classes=20, anchor_t=4, num_acnhors=3):
        super(TaskNearestAssigner, self).__init__()
        self.na = num_acnhors
        self.num_classes = num_classes
        self.topk = topk
        self.anchor_t = anchor_t

    @torch.no_grad()
    def forward(self, anc_wh, grid, gt_labels, gt_cxys, gt_whs, mask_gt):
        self.bs = gt_labels.shape[0]
        self.n_max_boxes = gt_labels.shape[1]

        if self.n_max_boxes == 0:
            return (
                None,
                None,
                None,
                torch.zeros(1, dtype=torch.bool, device=gt_labels.device),
            )
        # 获取真实目标的mask（重叠）
        # mask_pos: 样本（重叠）mask
        mask_pos, distance_metric = self.get_pos_mask(grid, gt_cxys, mask_gt)

        # 获取真实目标的mask、id（非重叠）
        # mask_pos: 样本（非重叠）mask (b, n_max_box, h*w)
        # fg_mask: 样本（非重叠）mask (b, h*w)
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos,
                                                                        distance_metric,
                                                                        self.n_max_boxes)
        # 制作标签
        target_txys, target_whs, target_scores = self.get_targets(gt_labels,
                                                                  gt_cxys,
                                                                  gt_whs,
                                                                  grid,
                                                                  target_gt_idx.unsqueeze(1).expand(-1, self.na, -1))
        # 剔除anchor不符合iou要求的正样本
        anc_wh = anc_wh.view(1, self.na, 1, -1)
        r = target_whs / anc_wh
        mask_anc = torch.max(r, 1 / r).max(-1)[0] < self.anchor_t
        fg_mask = fg_mask.unsqueeze(1) * mask_anc

        target_bboxes = torch.cat([target_txys, target_whs], -1)

        return target_bboxes, target_scores, anc_wh, fg_mask.bool()

    def get_pos_mask(self, grid, gt_cxys, mask_gt):
        # 计算真实目标中心点与网格中心点的距离
        distance_deltas = self.get_box_metrics(grid, gt_cxys)
        distance_metric = distance_deltas.abs().sum(-1)

        # 选取真实目标最近的k个网格mask
        mask_topk = self.select_topk_candidates(distance_metric, largest=False)

        # 真实目标框mask= 真实目标的k个目标mask * 非填充目标mask
        mask_pos = mask_topk * mask_gt

        return mask_pos, distance_metric

    def get_box_metrics(self, grid, gt_cxys):
        ng = grid.shape[0]
        gt_cxys = gt_cxys.view(-1, 1, 2)
        distance_deltas = ((grid[None] + 0.5) - gt_cxys).view(self.bs, self.n_max_boxes, ng, -1)
        return distance_deltas

    def select_topk_candidates(self, metrics, largest=True):
        # 每个bbox选取k个网格作为正样本
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)

        # 获取每个bbox的k个网格的mask
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=metrics.device)
        count_tensor.scatter_(-1, topk_idxs, 1)

        return count_tensor.to(metrics.dtype)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        # mask_pos: (b, n_max_boxes, h*w) -> (b, h*w)
        # 将每个网格的bbox合并一起，统计一个网格的bbox数量
        fg_mask = mask_pos.sum(-2)
        # 至少有一个重叠目标
        if fg_mask.max() > 1:
            # fg_mask: (b, 1, h*w) -> (b, n_max_boxes, h*w)
            mask_multi_gts = (fg_mask.unsqueeze(-2) > 1).expand(-1, n_max_boxes, -1)

            # 选取网格中距离最小的目标
            min_overlaps_idx = overlaps.argmin(-2)

            # non_overlaps: (b, n_max_boxes, h*w)
            non_overlaps = torch.ones(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)

            # 重叠网格全部置为 0
            non_overlaps[mask_multi_gts] = 0

            # 重叠网格最大分数置置为 1
            non_overlaps.scatter_(-2, min_overlaps_idx.unsqueeze(-2), 1)

            # 真实目标的mask = 真实目标（非重叠）* 非填充目标 -> (b, n_max_boxes, h*w)
            mask_pos = non_overlaps * mask_pos

            # 真实目标的mask (b, h*w)
            fg_mask = mask_pos.sum(-2)

        # 真实目标的id (b, h*w)
        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos

    def get_targets(self, gt_labels, gt_cxys, gt_whs, grid, target_gt_idx):
        ng = grid.shape[0]
        # batch idx: (b, 1, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None, None]
        # batch_ind * self.n_max_boxes: [0, 1*n, 2*n, ..., b*n]
        # target_gt_idx + (batch_ind * self.n_max_boxes):
        # 图1[目标1的id设置在0, ..., 目标n的id设置在n-1]
        # 图2[目标1的id设置在n, ..., 目标n的id设置在2n]
        # target_gt_idx (b, na, h*w)
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        # gt_labels: (b, n, 1) -> (b*n)
        # target_labels: (b, na, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]

        # gt_cxys: (b, na, n, 2) -> (b*na*n, 2)
        # target_cxys: (b, na, h*w, 2)
        target_cxys = gt_cxys.view(-1, gt_cxys.shape[-1])[target_gt_idx]
        target_txys = target_cxys - grid
        # gt_whs: (b, na, n, 2) -> (b*na*n, 2)
        # target_whs: (b, na, h*w, 2)
        target_whs = gt_whs.view(-1, gt_whs.shape[-1])[target_gt_idx]

        # target_scores: (b, na, h*w, c), One-Hot
        target_scores = torch.zeros((self.bs, self.na, ng, self.num_classes),
                                    dtype=torch.float,
                                    device=target_labels.device)  # (b, h*w, 80)
        # target_labels.unsqueeze(-1): (b, na, h*w,1)
        target_scores.scatter_(-1, target_labels.unsqueeze(-1), 1)

        return target_txys, target_whs, target_scores
