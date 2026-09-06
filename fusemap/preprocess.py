"""Compatibility shim: this module moved to fusemap.data.graph.

The alias below makes the old and new import paths point to the SAME module
object, so both `from fusemap.preprocess import *` and explicit (including
underscore-private) imports keep working unchanged.
"""
import sys as _sys

from fusemap.data.graph import *  # noqa: F401,F403 - ensures submodule import machinery runs
import fusemap.data.graph as _target

_sys.modules[__name__] = _target
