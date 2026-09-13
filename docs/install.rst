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

Google Colab
--------------------------------------------------------------------------------

Current Colab images can start with Python 3.13. FuseMap 1.1.3 still requires
Python 3.9–3.11 because it uses torch 2.0.1 and DGL 1.1.1. Installing with an
unrestricted ``pip install fusemap`` on a newer Python can select an obsolete
0.0.x release instead.

The :doc:`tutorial notebooks <tutorials>` include two setup cells:

1. Select **Runtime → Change runtime type → T4 GPU** before setup. Run the
   first cell to prepare Python 3.11 if the current Python is unsupported.
   It uses a pinned development revision of
   `CondaColab <https://github.com/conda-incubator/condacolab/tree/9df6578d7547f748e22d16b3a5755290bb41b9ad>`_
   that switches the actual notebook kernel, then restarts the session.
2. Wait for Colab to reconnect. Run the first cell again and confirm it reports
   Python 3.11. Run the install cell next. It installs ``fusemap[tutorials]==1.1.3``
   and, on a GPU runtime, the matching ``dgl==1.1.1+cu117`` wheel. The cell checks
   that the GPU graph backend works before you download tutorial data.
3. Continue with the tutorial when the cell prints ``Ready: Python ...``.

Setup takes several minutes and needs to be repeated for a fresh Colab runtime.
The temporary disconnect during the Python switch is expected. If setup fails
partway through, use **Runtime → Disconnect and delete runtime**, reconnect,
and start with the first setup cell. Simply restarting an unchanged Python 3.13
runtime or changing the shell's ``python`` command will not fix the notebook's
Python version.

The stable CondaColab 0.1.x package does not expose the ``python_version`` API
used here; keep the pinned source URL in the setup cell. The kernel environment
pins NumPy 1.26 and Matplotlib 3.8 before startup to match the older scientific
stack used by FuseMap and Scanpy 1.9.3.

For local Jupyter, use the Python 3.9–3.11 environment described above. The
tutorial setup cells leave a supported local environment in place. To use the
Colab interface with your own machine instead, follow Colab's
`local runtime instructions <https://research.google.com/colaboratory/local-runtimes.html>`_
with that environment. Memory and GPU requirements still depend on the chosen
tutorial dataset; the runtime setup does not download the data or run training.

Downloading the pretrained models
--------------------------------------------------------------------------------

You can download the following pretrained FuseMap models from
here: https://drive.google.com/drive/folders/1auybpmekWuW_G-7YPloJr-B96qiT1nFS?usp=sharing and put in local directories.


