Installation
============

.. attention::
    Our tool could be installed on Linux/Windows with Python3.8.

Anaconda
---------

We strongly recommend your operation in an isolated conda environment, so firstly run:

.. code-block:: 

    conda create --name st python=3.8  # The env name could be set arbitrarily, not only st.

Then get into the environment you build:

.. code-block:: 

    conda activate st

Use the installation command with conda:

.. code-block:: 

    conda install stereopy -c stereopy -c grst -c numba -c conda-forge -c bioconda

PyPI
----

The same beginning as conda part:

.. code-block:: 
    
    conda create --name st python=3.8

    conda activate st


Use PyPI run:

.. code-block:: 

    pip install stereopy

Development Version
--------------------

The same beginning as conda part:

.. code-block:: 

    conda create --name st python=3.8

    conda activate st


Use the latest version of dev branch on Github, you need to clone the repository and enter the directory: 

.. code-block:: 

    git clone -b dev https://github.com/STOmics/stereopy.git

    cd stereopy

    python setup.py install


Troubleshooting 
----------------

Possible installation failed due to some factors:

    Version of Python

Make sure you are working on Python3.8.

    Conflicts of dependencies

Find out packages that lead to failures, then create a new requirements.txt of them and run:

.. code-block:: 

    pip install -r requirements.txt


