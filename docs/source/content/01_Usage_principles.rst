Usage Principles 
================

Import Stereopy as:

.. code-block:: python

   import stereo as st

Workflow
---------

.. image:: ./../_static/Stereopy_workflow_v0.11.0.png
    :alt: Title figure
    :width: 700px
    :align: center

StereoExpData
--------------

**StereoExpData** is purposefully designed for express matrix of spatial omics, \
which displays high running performance in spite of huge data volume. \
It contains four important attributes, recording gene, cell, expression \
matrix and spatial location information respectively. The expression matrix \
supports both sparse and dense data. 