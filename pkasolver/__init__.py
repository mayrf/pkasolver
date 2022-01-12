"""
pkasolver
toolset for predicting the pka values of small molecules
"""

# Add imports here
from .pkasolver import *
from .dimorphite_dl.dimorphite_dl import run_with_mol_list

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions

import logging

# format logging message
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)1s()] %(message)s"
# set logging level
logging.basicConfig(format=FORMAT, datefmt="%d-%m-%Y:%H:%M", level=logging.INFO)
