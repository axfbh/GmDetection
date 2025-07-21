import glob
from pathlib import Path

from gmdet.utils import ROOT


def check_suffix(file="yolo11n.pt", suffix=".pt", msg=""):
    """
    Check file(s) for acceptable suffix.

    Args:
        file (str | List[str]): File or list of files to check.
        suffix (str | Tuple[str]): Acceptable suffix or tuple of suffixes.
        msg (str): Additional message to display in case of error.
    """
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix,)
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}, not {s}"


def check_file(file, suffix="", hard=True):
    """
    Search/download file (if necessary) and return path.

    Args:
        file (str): File name or path.
        suffix (str): File suffix to check.
        hard (bool): Whether to raise an error if the file is not found.

    Returns:
        (str): Path to the file.
    """
    check_suffix(file, suffix)  # optional
    file = str(file).strip()  # convert to string and strip spaces
    if (
            not file
            or ("://" not in file and Path(file).exists())  # '://' check required in Windows Python<3.10
            or file.lower().startswith("grpc://")
    ):  # file exists or gRPC Triton images
        return file
    else:  # search
        files = glob.glob(str(ROOT / "**" / file), recursive=True) or glob.glob(str(ROOT.parent / file))  # find file
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
        return files[0] if len(files) else []  # return file


def check_class_names(names):
    """Check class names and convert to dict format if needed."""
    if isinstance(names, list):  # names is a list
        names = dict(enumerate(names))  # convert to dict
    if isinstance(names, dict):
        # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
                f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
            )
    return names


def check_yaml(file, suffix=(".yaml", ".yml"), hard=True):
    """
    Search/download YAML file (if necessary) and return path, checking suffix.

    Args:
        file (str): File name or path.
        suffix (tuple): Acceptable file suffixes.
        hard (bool): Whether to raise an error if the file is not found.

    Returns:
        (str): Path to the YAML file.
    """
    return check_file(file, suffix, hard=hard)
