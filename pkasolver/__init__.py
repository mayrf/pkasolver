"""
pkasolver
toolset for predicting the pka values of small molecules
"""

# Add imports here
from pkasolver import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
