import torch
import torchvision
from torchvision.ops.boxes import box_convert

torch.set_printoptions(precision=4, sci_mode=False)


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_wh=7680,
        max_nms=30000,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a modules, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the modules is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the modules. Any indices after this will be considered masks.
        max_nms (int): The maximum number of boxes into torchvision.utils.nms().
        max_wh (int): The maximum box width and height in pixels.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction,
                  (list, tuple)):  # YOLOv8 modules in validation modules, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    bs, c1, c2 = prediction.shape
    is_v8 = c1 < c2

    # v8后的版本
    if is_v8:
        prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
        nc = nc or (prediction.shape[2] - 4)  # number of classes
        nm = prediction.shape[2] - nc - 4  # mask start index
        mi = 4 + nc
    else:
        nc = nc or (prediction.shape[2] - 5)  # number of classes
        nm = prediction.shape[2] - nc - 5  # mask start index
        mi = 5 + nc

    # v8后的版本
    if is_v8:
        xc = prediction[..., 4:mi].amax(-1) > conf_thres  # candidates
    else:
        xc = prediction[..., 4] > conf_thres  # candidates

    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    box = box_convert(prediction[..., :4], in_fmt='cxcywh', out_fmt='xyxy')
    prediction = torch.cat((box, prediction[..., 4:]), dim=-1)  # xywh to xyxy

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            # v8后的版本
            if is_v8:
                v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            else:
                v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            # v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[:, :4] = box_convert(lb[:, 1:5], in_fmt='cxcywh', out_fmt='xyxy')  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        if is_v8:
            box, cls, mask = x.split((4, nc, nm), 1)
        else:
            box, obj, cls, mask = x.split((4, 1, nc, nm), 1)
            # Compute conf
            cls *= obj

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = torch.where(cls > conf_thres)
            # v8后的版本
            if is_v8:
                x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:
                x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float() + 1, mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            # v8后的版本
            if is_v8:
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
            else:
                x = torch.cat((box, conf, j.float() + 1, mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue

        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        boxes = x[:, :4] + c  # boxes (offset by class)
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        output[xi] = x[i]

    return output
