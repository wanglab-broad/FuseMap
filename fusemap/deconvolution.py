"""Compatibility shim: this module moved to fusemap.postprocess.imputation.

The alias below makes the old and new import paths point to the SAME module
object, so both `from fusemap.deconvolution import *` and explicit (including
underscore-private) imports keep working unchanged.
"""
import sys as _sys

from fusemap.postprocess.imputation import *  # noqa: F401,F403 - ensures submodule import machinery runs
import fusemap.postprocess.imputation as _target

_sys.modules[__name__] = _target
