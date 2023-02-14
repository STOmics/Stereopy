Installation
============
.. note::
    **Our tool could be installed on Linux with python3.8.**

PyPI
------------------------------------

**Step1**:

Prepare an isolated conda environment

.. code:: bash

    conda create --name st python=3.8
    conda activate st

**Step2**:

Install Stereopy using *pip*

.. code:: bash

    pip install stereopy

Anaconda
------------------------------------
conda create -n your_stereopy_env

conda activate your_stereopy_env

conda install stereopy -c stereopy -c grst -c numba -c conda-forge -c bioconda

Development Version
------------------------------------
**Step1**:

Prepare an isolated conda environment

.. code:: bash

    conda create --name st python=3.8
    conda activate st

**Step2**:

If you want to use the latest version of dev branch on GitHub, you need to clone the repository and enter the directory.

.. code:: bash

    git clone -b dev https://github.com/BGIResearch/stereopy.git

    cd stereopy

    python setup.py install

