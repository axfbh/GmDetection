import os
import glob
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

import cv2
from skimage import io
from torch.utils.data import Dataset

from utils import DEFAULT_CFG, LOGGER
from data.utils import IMG_FORMATS, FORMATS_HELP_MSG, img2label_paths, verify_image_label


class BaseDataset(Dataset):
    def __init__(self, img_path, imgsz):
        self.img_path = img_path
        self.imgsz = imgsz
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()

    def get_img_files(self, img_path):
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                else:
                    raise FileNotFoundError(f"{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            assert im_files, f"No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"Error loading data") from e
        return im_files

    def cache_labels(self):
        labels = []
        nm, nf, ne = 0, 0, 0
        total = len(self.im_files)
        desc = f"Scanning ..."
        results = []
        for im_file, lb_file in zip(self.im_files, self.label_files):
            results.append(verify_image_label(im_file, lb_file))

        pbar = tqdm(results, desc=desc, total=total)

        for im_file, lb, segments, nm_f, nf_f, ne_f in pbar:
            nm += nm_f
            nf += nf_f
            ne += ne_f
            if im_file:
                if not segments:
                    labels.append({
                        "image": im_file,
                        "labels": lb[:, 0],  # n, 1
                        "bboxes": lb[:, 1:]})  # n, 4
                else:
                    labels.append({
                        "image": im_file,
                        "labels": lb[:, 0],  # n, 1
                        "segments": segments})

                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds"
        pbar.close()
        return labels

    def get_labels(self):
        self.label_files = img2label_paths(self.im_files)
        labels = self.cache_labels()

        if not labels:
            LOGGER.warning(f"No images found, training may not work correctly.")

        # Check if the dataset is all boxes or all segments
        len_cls = (len(lb["labels"]) for lb in labels)

        # 没有标签，可能存在错误
        if len_cls == 0:
            LOGGER.warning(f"No labels found, training may not work correctly.")

        return labels

    def load_image(self, f):
        im = cv2.cvtColor(io.imread(f), cv2.COLOR_RGB2BGR)
        return im

    def get_image_and_label(self, index):
        label = deepcopy(self.labels[index])
        img_path = label['image']
        img = self.load_image(img_path)
        label['image'] = img
        return label

    def __len__(self):
        """Return the length of the labels list for the dataset."""
        return len(self.labels)
