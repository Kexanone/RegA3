'''
    RegA3
    ~~~~~

    Export trained regression models from Scikit-learn to SQF.
'''

from .__version__ import __version__
from .__version__ import __author__

from . import exporter
from .exporter import export_estimator

__all__ = []
__all__ += exporter.__all__
