.. _Installation:

Installation
================================================================================

Installing the package
--------------------------------------------------------------------------------

The FuseMap package can be
downloaded from `GitHub <https://github.com/yichunher/fusemap>`__.
Installation is quick and performed using ``pip`` in the usual manner:

::

    conda create -n fusemap python=3.10
    conda activate fusemap
    pip install fusemap

.. note::

    A GPU is necessary for accelerating computations.
    Estimated time is 1 hours for integrating 300,000 cells on a single GPU.

Downloading the pretrained models
--------------------------------------------------------------------------------

You can download the following pretrained FuseMap models from
Zenodo and put in local directories.

Mouse brain: xx

Mouse and human organs: xx




.. Usage
.. =====

.. Installation
.. ------------

.. To use FuseMap, install from GitHub with pip:

.. .. code-block:: console

..    $ git clone https://github.com/yichunher/fusemap.git
..    $ cd scimilarity
..    $ pip install -e .


.. SpatialIntegration
.. ----------------

.. To retrieve a list of random ingredients,
.. you can use the ``fusemap.spatial_integrate()`` function:

.. .. py:function:: lumache.get_random_ingredients(kind=None)

..    Return a list of random ingredients as strings.

..    :param kind: Optional "kind" of ingredients.
..    :type kind: list[str] or None
..    :return: The ingredients list.
..    :rtype: list[str]


.. SpatialMapping
.. ----------------

.. To retrieve a list of random ingredients,
.. you can use the ``fusemap.spatial_map()`` function:

.. .. py:function:: lumache.get_random_ingredients(kind=None)

..    Return a list of random ingredients as strings.

..    :param kind: Optional "kind" of ingredients.
..    :type kind: list[str] or None
..    :return: The ingredients list.
..    :rtype: list[str]


