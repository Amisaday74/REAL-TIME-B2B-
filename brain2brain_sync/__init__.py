# b2b_synchrony/__init__.py

from .EEG_device import EEG
from .bispectrum import bispec
from .stopwatch import timer
from .graphs import Graph

__all__ = ["EEG", "bispec", "timer", "Graph"]