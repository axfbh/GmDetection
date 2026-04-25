#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple
from multiprocessing import Pool

def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def parse_selected_categories(coco: Dict, names: List[str], ids: List[int]) -> Set[int]:
    cat_name_to_id = {c["name"]: c["id"] for c in coco.get("categories", [])}
    all_ids = {c["id"] for c in coco.get("categories", [])}

    selected: Set[int] = set()

    # by name
    for n in names:
        if n not in cat_name_to_id:
            raise ValueError(f"类别名不存在: {n}")
        selected.add(cat_name_to_id[n])

    # by id
    for cid in ids:
        if cid not in all_ids:
            raise ValueError(f"类别ID不存在: {cid}")
        selected.add(cid)

    if not selected:
        raise ValueError("未选择任何类别，请传入 --cat-names 或 --cat-ids。")

    return selected


def build_subset(
    coco: Dict,
    selected_cat_ids: Set[int],
    keep_empty_images: bool = False
) -> Tuple[Dict, Set[int], int]:
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    # 1) 筛选目标类别的标注
    new_annotations = [ann for ann in annotations if ann["category_id"] in selected_cat_ids]

    # 2) 得到包含目标标注的图像ID
    image_ids_with_ann = {ann["image_id"] for ann in new_annotations}

    # 3) 筛选图像
    if keep_empty_images:
        # 保留所有图像（即使没有目标标注）
        new_images = images
        kept_image_ids = {img["id"] for img in images}
    else:
        # 仅保留有目标标注的图像
        new_images = [img for img in images if img["id"] in image_ids_with_ann]
        kept_image_ids = {img["id"] for img in new_images}

    # 4) 保留类别
    new_categories = [c for c in categories if c["id"] in selected_cat_ids]

    # 5) 如果 keep_empty_images=True，仍需确保 annotation 的 image_id 在保留图像里
    new_annotations = [ann for ann in new_annotations if ann["image_id"] in kept_image_ids]

    out = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": new_images,
        "annotations": new_annotations,
        "categories": new_categories,
    }
    return out, kept_image_ids, len(new_annotations)


def copy_images(
    image_root: Path,
    output_image_root: Path,
    images: List[Dict],
    skip_missing: bool = False
) -> None:
    output_image_root.mkdir(parents=True, exist_ok=True)
    missing = 0

    for img in images:
        file_name = img["file_name"]
        src = image_root / file_name
        dst = output_image_root / file_name
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            missing += 1
            if not skip_missing:
                raise FileNotFoundError(f"找不到图像文件: {src}")
            continue

        shutil.copy2(src, dst)

    if missing > 0:
        print(f"[警告] 有 {missing} 张图像未找到，已跳过。")


def main(ann, out_ann, cat_names, image_root, out_image_root, skip_missing=False):

    coco = load_json(ann)
    selected_cat_ids = parse_selected_categories(coco, cat_names, [])

    subset, kept_image_ids, ann_count = build_subset(
        coco, selected_cat_ids, keep_empty_images=False
    )

    save_json(subset, out_ann)

    print("=== 提取完成 ===")
    print(f"输入标注: {ann}")
    print(f"输出标注: {out_ann}")
    print(f"保留类别ID: {sorted(selected_cat_ids)}")
    print(f"图像数量: {len(subset['images'])}")
    print(f"标注数量: {ann_count}")

    if image_root and out_image_root:
        copy_images(image_root, out_image_root, subset["images"], skip_missing)
        print(f"图像已复制到: {out_image_root}")
    elif image_root or out_image_root:
        raise ValueError("如果要复制图像，--image-root 与 --out-image-root 必须同时提供。")

    print("=== 提取完成 ===")


if __name__ == "__main__":
    num_workers = 2
    root = Path("/mnt/f/dataset")

    cat_names = [["dog"], ["dog"]]
    skip_missing = [False, False]

    anns = [
        root.joinpath("coco2017","annotations","instances_train2017.json"),
        root.joinpath("coco2017","annotations","instances_val2017.json")
    ]
    out_anns = [
        root.joinpath("sub_coco_dog","annotations","instances_train2017.json"),
        root.joinpath("sub_coco_dog","annotations","instances_val2017.json")
    ]
    image_roots = [
        root.joinpath("coco2017","train2017"),
        root.joinpath("coco2017","val2017")
    ]
    out_image_roots = [
        root.joinpath("sub_coco_dog","train2017"),
        root.joinpath("sub_coco_dog","val2017")
    ]

    with Pool(num_workers) as p:
        p.starmap(main, zip(anns, out_anns, cat_names, image_roots, out_image_roots, skip_missing))

