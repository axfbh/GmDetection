import os
import shutil

import numpy as np
from pycocotools.coco import COCO
from pathlib import Path
from gmdet.utils.numpy_utils import xyxy_to_cxcywh


def change_2_yolo(json_file):
    coco = COCO(json_file)

    img_ids = coco.getImgIds()

    cats = coco.cats
    cats_sequence = {k: i for i, k in enumerate(cats.keys())}

    res = []
    for img_id in img_ids:
        targets = []

        image = coco.loadImgs(img_id)[0]
        height = image['height']
        width = image['width']
        file_name = image['file_name']
        ann_id = coco.getAnnIds(img_id)
        anns = coco.loadAnns(ann_id)

        for ann in anns:
            cat_id = ann['category_id']
            cat = int(cats_sequence[coco.loadCats(cat_id)[0]['id']])
            # x_min, y_min, width, height
            bbox = np.array(ann['bbox'], dtype=np.float32)
            bbox[2:] += bbox[:2]
            bbox = xyxy_to_cxcywh(bbox) / np.array([width, height, width, height])

            seg = (np.array(ann['segmentation'][0], dtype=np.float32).reshape(-1, 2) / np.array(
                [width, height])).reshape(-1)

            targets.append([cat, bbox])
            # targets.append([cat, seg])

        res.append([file_name, targets])

    return res


def image_txt_copy(files, scr_path, dst_img_path, dst_txt_path):
    for file in files:
        filename = Path(file[0])
        targets = file[1]
        img_path = os.path.join(scr_path, filename)
        shutil.copy(img_path, os.path.join(dst_img_path, filename))
        scr_txt_path = os.path.join(dst_txt_path, filename.stem + '.txt')
        with open(scr_txt_path, 'w') as wp:
            for tg in targets:
                wp.write(str(tg[0]) + " " + " ".join([str(a) for a in tg[1]]) + '\n')


if __name__ == '__main__':
    root = Path(r"D:\dataset\coco_sub_dog")
    modes = ['train', 'val']
    filenames = 'VOC'
    json_files = [root.joinpath("annotations", "instances_train2017.json"),
                  root.joinpath("annotations", "instances_val2017.json")]

    image_roots = [root.joinpath("train2017"),
                   root.joinpath("val2017")]

    for js_f, img_r, m in zip(json_files, image_roots, modes):
        train_list = change_2_yolo(js_f)

        if not os.path.exists(f'../../{filenames}/images/%s' % m):
            os.makedirs(f'../../{filenames}/images/%s' % m)

        if not os.path.exists(f'../{filenames}/labels/%s' % m):
            os.makedirs(f'../../{filenames}/labels/%s' % m)

        image_txt_copy(train_list, img_r, f'../../{filenames}/images/{m}/', f'../../{filenames}/labels/{m}/')
