.. _Installation:

Installation
================================================================================

Installing the package
--------------------------------------------------------------------------------

The FuseMap package can be
downloaded from `GitHub <https://github.com/wanglab-broad/FuseMap>`__.
Installation is quick and performed using ``pip`` in the usual manner:

::

    conda create -n fusemap python=3.10
    conda activate fusemap
    pip install fusemap

.. note::

    Supported Python versions: 3.9-3.11 (the pinned torch 2.0.1 / dgl 1.1.1
    wheels are not available for 3.12+). A GPU is necessary for accelerating computations.
    Estimated time is 10 mins for integrating 200,000 cells with a single GPU.

Installing from a checkout
--------------------------------------------------------------------------------

For the version in this repository (including unreleased fixes):

.. code-block:: bash

    git clone https://github.com/wanglab-broad/FuseMap.git
    cd FuseMap
    python -m pip install .

Install notebook dependencies with ``python -m pip install ".[tutorials]"``.
``requirements.txt`` installs the same checkout using the dependencies in
``setup.py``. ``fusemap_environment.yaml`` is a historical Python 3.7 environment
and is not the installation specification for this release.

The wheel provides the Python API. ``main.py``, the Agent interfaces, and the
notebooks are used from the repository checkout. See :doc:`agent/index` for
classic Agent dependencies and the separate Agent v2 environment.

Downloading the pretrained models
--------------------------------------------------------------------------------

You can download the following pretrained FuseMap models from
here: https://drive.google.com/drive/folders/1auybpmekWuW_G-7YPloJr-B96qiT1nFS?usp=sharing and put in local directories.



