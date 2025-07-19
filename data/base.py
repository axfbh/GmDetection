import os
import glob
from pathlib import Path
from copy import deepcopy

import cv2
from skimage import io
from torch.utils.data import Dataset

from data import IMG_FORMATS, FORMATS_HELP_MSG


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

    def load_image(self, f):
        im = cv2.cvtColor(io.imread(f), cv2.COLOR_RGB2BGR)
        return im

    def get_image_and_label(self, index):
        label = deepcopy(self.labels[index])
        img_path = label['image']
        img = self.load_image(img_path)
        label['image'] = img
        return label

    def get_labels(self):
        """Users can customize their own format here."""
        raise NotImplementedError

    def __len__(self):
        """Return the length of the labels list for the dataset."""
        return len(self.labels)
