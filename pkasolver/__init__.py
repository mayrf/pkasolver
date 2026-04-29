"""
pkasolver
Toolkit for predicting microstate pKa values of small molecules
via Graph Isomorphism Networks (GINs).
"""

import logging

from .dimorphite_dl.dimorphite_dl import run_with_mol_list  # noqa: F401

try:
    from ._version import __version__
except ImportError:
    # Package not installed (e.g. running from source without build)
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "run_with_mol_list",
]

# Configure logging
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(format=FORMAT, datefmt="%d-%m-%Y:%H:%M", level=logging.WARNING)
