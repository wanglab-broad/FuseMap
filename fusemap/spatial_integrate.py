"""Compatibility shim: this module moved to fusemap.training.integrate.

The alias below makes the old and new import paths point to the SAME module
object, so both `from fusemap.spatial_integrate import *` and explicit (including
underscore-private) imports keep working unchanged.
"""
import sys as _sys

from fusemap.training.integrate import *  # noqa: F401,F403 - ensures submodule import machinery runs
import fusemap.training.integrate as _target

_sys.modules[__name__] = _target
