"""Kernel ridge regression with the Volterra reservoir kernel."""

from ._estimator import VolterraKernelRidge
from ._kernel import volterra_kernel, zero_history_value

__all__ = ["VolterraKernelRidge", "volterra_kernel", "zero_history_value"]
__version__ = "0.1.0"
