"""Compatibility shim: this module moved to fusemap.training.train_model.

The alias below makes the old and new import paths point to the SAME module
object, so both `from fusemap.train_model import *` and explicit (including
underscore-private) imports keep working unchanged.
"""
import sys as _sys

from fusemap.training.train_model import *  # noqa: F401,F403 - ensures submodule import machinery runs
import fusemap.training.train_model as _target

_sys.modules[__name__] = _target
