import re
import os.path
import yaml
from pathlib import Path
from omegaconf import OmegaConf

import torch
from torch import nn
import socket


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt = torch.load(weight)  # load ckpt
    model = ckpt.get("ema").to(device).float()  # FP32 model

    # # Module updates
    # for m in model.modules():
    #     if hasattr(m, "inplace"):
    #         m.inplace = inplace
    #     elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
    #         m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    model = os.path.splitext(unified_path)[0]
    v = model[4:]
    if path.is_file():
        yaml_file = path
    else:
        yaml_file = os.path.join(f'./cfg/models/{v}', unified_path)
    d = OmegaConf.load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    d["model"] = model
    return d


def yaml_save(file="data.yaml", data=None, header=""):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict): Data to save in YAML format.
        header (str, optional): YAML header to add.

    Returns:
        (None): Data is saved to the specified file.
    """
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings
    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)

    # Dump data to file in YAML format
    with open(file, "w", errors="ignore", encoding="utf-8") as f:
        if header:
            f.write(header)
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale. The function
    uses regular expression matching to find the pattern of the model scale in the YAML file name, which is denoted by
    n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    try:
        return re.search(r"yolo[v]?\d+([nslmx])", Path(model_path).stem).group(1)  # noqa, returns n, s, m, l, or x
    except AttributeError:
        return ""


def ip_load():
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        st.connect(('10.255.255.255', 1))
        IP = st.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        st.close()
    return IP
