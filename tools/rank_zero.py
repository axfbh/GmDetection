import logging
from functools import wraps
from typing import Callable, Optional, TypeVar, Union, Any
from platform import python_version

from typing_extensions import ParamSpec, overload

log = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


def rank_zero_only(fn: Callable[P, T], default: Optional[T] = None) -> Callable[P, Optional[T]]:
    """Wrap a function to call internal function only in rank zero.

    Function that can be used as a decorator to enable a function/method being called only on global rank 0.

    """

    @wraps(fn)
    def wrapped_fn(*args: P.args, **kwargs: P.kwargs) -> Optional[T]:
        rank = getattr(rank_zero_only, "rank", None)
        if rank is None:
            raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
        if rank == 0:
            return fn(*args, **kwargs)
        return default

    return wrapped_fn


def _info(*args: Any, stacklevel: int = 2, **kwargs: Any) -> None:
    if python_version() >= "3.8.0":
        kwargs["stacklevel"] = stacklevel
    log.info(*args, **kwargs)


@rank_zero_only
def rank_zero_info(*args: Any, stacklevel: int = 4, **kwargs: Any) -> None:
    """Emit info-level messages only on global rank 0."""
    _info(*args, stacklevel=stacklevel, **kwargs)


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    if input[0] == 'Warning:':
        *args, string = ("red", "bold", input[0])
    elif len(input) > 1:
        *args, string = input
    else:
        *args, string = ("blue", "bold", input[0])  # color arguments, string

    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]
