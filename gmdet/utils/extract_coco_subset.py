#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple


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


def main():
    # parser = argparse.ArgumentParser(description="从COCO标注中提取指定类别子集")
    # parser.add_argument("--ann", required=True, type=Path, help="输入COCO标注json路径")
    # parser.add_argument("--out-ann", required=True, type=Path, help="输出子集json路径")
    # parser.add_argument("--cat-names", nargs="*", default=[], help="按类别名筛选，例如 person car")
    # parser.add_argument("--cat-ids", nargs="*", default=[], type=int, help="按类别ID筛选，例如 1 3 17")
    # parser.add_argument("--keep-empty-images", action="store_true", help="保留无目标标注的图像")
    # parser.add_argument("--image-root", type=Path, default=None, help="原图根目录(可选)")
    # parser.add_argument("--out-image-root", type=Path, default=None, help="输出图根目录(可选)")
    # parser.add_argument("--skip-missing", action="store_true", help="复制图像时跳过缺失文件")
    # args = parser.parse_args()

    ann = Path("/mnt/f/dataset/coco2017/annotations/instances_train2017.json")
    out_ann = Path("/mnt/f/dataset/sub_coco_dog/annotations/instances_train2017.json")
    cat_names = ["dog"]
    image_root = Path("/mnt/f/dataset/coco2017/train2017")
    out_image_root = Path("/mnt/f/dataset/sub_coco_dog/train2017")
    skip_missing = False

    # ann = Path("/mnt/f/dataset/coco2017/annotations/instances_val2017.json")
    # out_ann = Path("/mnt/f/dataset/sub_coco_dog/annotations/instances_val2017.json")
    # cat_names = ["dog"]
    # image_root = Path("/mnt/f/dataset/coco2017/val2017")
    # out_image_root = Path("/mnt/f/dataset/sub_coco_dog/val2017")
    # skip_missing = False

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
        copy_images(
            image_root=image_root,
            output_image_root=out_image_root,
            images=subset["images"],
            skip_missing=skip_missing
        )
        print(f"图像已复制到: {out_image_root}")
    elif image_root or out_image_root:
        raise ValueError("如果要复制图像，--image-root 与 --out-image-root 必须同时提供。")


if __name__ == "__main__":
    main()