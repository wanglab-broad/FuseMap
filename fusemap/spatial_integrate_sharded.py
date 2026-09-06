"""Compatibility shim: this module moved to fusemap.training.sharded.

The alias below makes the old and new import paths point to the SAME module
object, so both `from fusemap.spatial_integrate_sharded import *` and explicit (including
underscore-private) imports keep working unchanged.
"""
import sys as _sys

from fusemap.training.sharded import *  # noqa: F401,F403 - ensures submodule import machinery runs
import fusemap.training.sharded as _target

_sys.modules[__name__] = _target
