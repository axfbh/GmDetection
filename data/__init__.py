import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from utils.numpy_utils import xyxy_to_cxcywh, cxcywh_to_xyxy

IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}  # image suffixes
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}"


def exif_size(img: Image.Image):
    """Returns exif-corrected PIL size."""
    s = img.size  # (width, height)
    if img.format == "JPEG":  # only support JPEG images
        try:
            if exif := img.getexif():
                rotation = exif.get(274, None)  # the EXIF key for the orientation tag is 274
                if rotation in {6, 8}:  # rotation 270 or 90
                    s = s[1], s[0]
        except Exception:
            pass
    return s


def segments2boxes(segments):
    """
    Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh).

    Args:
        segments (list): List of segments, each segment is a list of points, each point is a list of x, y coordinates.

    Returns:
        (np.ndarray): The xywh coordinates of the bounding boxes.
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy_to_cxcywh(np.array(boxes))  # cls, xywh


def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def verify_image_label(im_file, lb_file, num_cls):
    """Verify one image-label pair."""
    # Number (missing, found, empty), message, segments, keypoints
    nm, nf, ne, segments = 0, 0, 0, []

    # Verify images
    im = Image.open(im_file)
    im.verify()  # PIL verify
    shape = exif_size(im)  # image size
    shape = (shape[1], shape[0])  # hw
    assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
    assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}. {FORMATS_HELP_MSG}"
    if im.format.lower() in {"jpg", "jpeg"}:
        with open(im_file, "rb") as f:
            f.seek(-2, 2)
            if f.read() != b"\xff\xd9":  # corrupt JPEG
                ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)

    # Verify labels
    if os.path.isfile(lb_file):
        nf = 1  # label found
        with open(lb_file, encoding="utf-8") as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]

            # 判断是不是分割标签
            if any(len(x) > 6 for x in lb):  # is segment
                classes = np.array([x[0] for x in lb], dtype=np.float32)
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)

            lb = np.array(lb, dtype=np.float32)

        # 标签文件有标签内容
        if nl := len(lb):
            # bbox 标签 长度为5 (cls, xywh)
            assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
            points = lb[:, 1:]
            # 必须是归一化标签
            assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
            assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

            max_cls = lb[:, 0].max()

            assert max_cls < num_cls, (
                f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                f"Possible class labels are 0-{num_cls - 1}"
            )

            # 类别
            _, i = np.unique(lb, axis=0, return_index=True)

            # 多少个标签的类别相同
            if len(i) < nl:  # duplicate row check
                lb = lb[i]  # remove duplicates
                if segments:
                    segments = [segments[x] for x in i]
        else:
            ne = 1  # label empty
            lb = np.zeros((0, 5), dtype=np.float32)
    else:
        nm = 1  # label missing
        lb = np.zeros((0, 5), dtype=np.float32)
    lb = lb[:, :5]
    return im_file, lb, segments, nm, nf, ne


def visualize_bbox(batch, color=(255, 0, 0), thickness=2):
    img = batch['image']
    bbox = batch['bboxes']
    h, w, c = img.shape

    """Visualizes a single bounding box on the image"""
    for box in bbox:
        x_min, y_min, x_max, y_max = cxcywh_to_xyxy(box)

        x_min = int(x_min * w)
        y_min = int(y_min * h)
        x_max = int(x_max * w)
        y_max = int(y_max * h)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    from PIL import Image
    Image.fromarray(img).show()
