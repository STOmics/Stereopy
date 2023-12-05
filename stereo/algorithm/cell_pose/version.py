"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from platform import python_version

import torch

try:
    version = version("cellpose")
except PackageNotFoundError:
    version = 'unknown'

version_str = f"""
cellpose version: \t{version}
platform:       \t{sys.platform}
python version: \t{python_version()}
torch version:  \t{torch.__version__}"""
