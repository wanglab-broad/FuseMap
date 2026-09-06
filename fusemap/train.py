"""Compatibility shim: this module moved to fusemap.training.cli.

The alias below makes the old and new import paths point to the SAME module
object, so both `from fusemap.train import *` and explicit (including
underscore-private) imports keep working unchanged.
"""
import sys as _sys

from fusemap.training.cli import *  # noqa: F401,F403 - ensures submodule import machinery runs
import fusemap.training.cli as _target

_sys.modules[__name__] = _target
