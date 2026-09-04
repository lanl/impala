from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version

from . import physics, superCal

try:
    __version__ = _get_version("impala-calib")
except PackageNotFoundError:
    __version__ = (
        "local"  ## use this if impala is not installed,  but imported locally
    )

__all__ = ["physics", "superCal"]
