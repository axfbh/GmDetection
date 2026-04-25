# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
from gmdet.models.yolo.utils.iou_loss import all_iou_loss


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
    """基于“最近网格点”的 Anchor-Based 分配器。"""

    def __init__(self, topk=3, num_classes=20, anchor_t=4, num_anchors=3):
        super(TaskNearestAssigner, self).__init__()
        self.na = num_anchors
        self.num_classes = num_classes
        self.topk = topk
        self.anchor_t = anchor_t

    @torch.no_grad()
    def forward(self, anc_wh, grid, gt_labels, gt_cxys, gt_whs, mask_gt):
        """
        根据 GT 中心点与网格中心距离构建训练目标。
        Args:
            anc_wh: (1, na, 1, 2)  anchor 宽高 [w, h]
            grid: (h*w, 2) 网格中心点 [x, y]
            gt_labels: (b, n, 1)  GT 类别标签   
            gt_cxys: (b, n, 2)  GT 中心点 [x, y]
            gt_whs: (b, n, 2)  GT 宽高  [w, h]
            mask_gt: (b, max_num_obj, h*w) 非填充样本 mask

        Returns:
            target_bboxes: (b, na, h*w, 4), 格式为 [tx, ty, w, h]
            target_scores: (b, na, h*w, num_classes), one-hot 分类标签
            anc_wh: (1, na, 1, 2), 变形后的 anchor 宽高
            fg_mask: (b, na, h*w), 正样本掩码
        """
        self.bs = gt_labels.shape[0]
        self.n_max_boxes = gt_labels.shape[1]

        if self.n_max_boxes == 0:
            return (
                None,
                None,
                None,
                torch.zeros(1, dtype=torch.bool, device=gt_labels.device),
            )

        # 1) 候选正样本：每个 GT 选择最近的 top-k 网格。
        mask_pos, distance_metric = self.get_pos_mask(grid, gt_cxys, mask_gt)

        # 2) 冲突消解：同一网格若匹配多个 GT，仅保留距离最小者。
        target_gt_idx, fg_mask, _ = self.select_highest_overlaps(mask_pos, distance_metric, self.n_max_boxes)

        # 3) 依据分配结果构建 tx/ty/wh 与 one-hot 分类标签。
        expanded_target_gt_idx = target_gt_idx.unsqueeze(1).expand(-1, self.na, -1)
        target_txys, target_whs, target_scores = self.get_targets(
            gt_labels, gt_cxys, gt_whs, grid, expanded_target_gt_idx
        )

        # 4) 根据 anchor 与 GT 宽高比例过滤不匹配正样本。
        anc_wh = anc_wh.view(1, self.na, 1, -1)
        r = target_whs / anc_wh
        mask_anc = torch.max(r, 1 / r).max(-1)[0] < self.anchor_t
        fg_mask = fg_mask.unsqueeze(1) * mask_anc

        target_bboxes = torch.cat([target_txys, target_whs], -1)

        return target_bboxes, target_scores, anc_wh, fg_mask.bool()

    def get_pos_mask(self, grid, gt_cxys, mask_gt):
        """生成候选正样本掩码与距离度量。"""
        # 计算 GT 中心点与网格中心点的曼哈顿距离。
        distance_deltas = self.get_box_metrics(grid, gt_cxys)
        distance_metric = distance_deltas.abs().sum(-1)

        # 每个 GT 选择最近 top-k 个网格。
        mask_topk = self.select_topk_candidates(distance_metric, largest=False)

        # 仅保留非填充 GT 对应位置。
        mask_pos = mask_topk * mask_gt

        return mask_pos, distance_metric

    def get_box_metrics(self, grid, gt_cxys):
        """计算 GT 中心点相对网格中心点的偏移量。"""
        ng = grid.shape[0]
        gt_cxys = gt_cxys.view(-1, 1, 2)
        distance_deltas = ((grid[None] + 0.5) - gt_cxys).view(self.bs, self.n_max_boxes, ng, -1)
        return distance_deltas

    def select_topk_candidates(self, metrics, largest=True):
        """每个 GT 选 top-k 网格，返回 0/1 掩码。"""
        _, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)

        # 记录每个 GT 的 top-k 网格位置。
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=metrics.device)
        count_tensor.scatter_(-1, topk_idxs, 1)

        return count_tensor.to(metrics.dtype)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """当一个网格命中多个 GT 时，仅保留距离最近的 GT。"""
        # 统计每个网格被多少个 GT 选中，shape: (b, h*w)。
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            # 标出被多个 GT 同时命中的网格。
            mask_multi_gts = (fg_mask.unsqueeze(-2) > 1).expand(-1, n_max_boxes, -1)

            # 选取网格中距离最小的目标
            min_overlaps_idx = overlaps.argmin(-2)

            # 非重叠网格全部置为 1
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
        """按匹配下标收集训练目标。"""
        ng = grid.shape[0]
        # target_gt_idx: (b, na, h*w)，每个位置存的是“匹配到的 GT 下标（0~n-1）”。
        # 下面统一用 gather：先把 GT 张量扩展出 anchor 维，再按 target_gt_idx 在 GT 维(n)上取值。   

        # 1) 分类标签
        # gt_labels: (b, n, 1) -> (b, n) -> (b, na, n)
        labels_src = gt_labels.squeeze(-1).long().unsqueeze(1).expand(-1, self.na, -1)
        # 在 dim=2（GT 维）按 target_gt_idx 取值，得到 (b, na, h*w)
        target_labels = torch.gather(labels_src, dim=2, index=target_gt_idx)

        # 2) 中心点坐标
        # gt_cxys: (b, n, 2) -> (b, na, n, 2)
        cxys_src = gt_cxys.unsqueeze(1).expand(-1, self.na, -1, -1)
        # gather 的 index 需要和输出同维，故补上最后一维并扩展到 2
        gather_idx = target_gt_idx.unsqueeze(-1).expand(-1, -1, -1, gt_cxys.shape[-1])
        # 在 dim=2（GT 维）取值，得到 (b, na, h*w, 2)
        target_cxys = torch.gather(cxys_src, dim=2, index=gather_idx)
        target_txys = target_cxys - grid

        # 3) 宽高
        # gt_whs: (b, n, 2) -> (b, na, n, 2)
        whs_src = gt_whs.unsqueeze(1).expand(-1, self.na, -1, -1)
        target_whs = torch.gather(whs_src, dim=2, index=gather_idx)

        # one-hot 分类目标: (b, na, h*w, c)
        target_scores = torch.zeros((self.bs, self.na, ng, self.num_classes),
                                    dtype=torch.float,
                                    device=target_labels.device)
        target_scores.scatter_(-1, target_labels.unsqueeze(-1), 1)

        return target_txys, target_whs, target_scores
