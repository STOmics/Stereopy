Installation
============
.. note::
    Our tool could be installed on Linux with python3.7 or python3.8.

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

Not yet.

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

Popular bugs
------------------------------------

Installation failed due to some factors:

**Version of Python**

    make sure you are using python3.7 or python3.8

**Conficts of dependencies**

    find out packages which lead to failures

    create a new requirements.txt and run:

.. code:: bash

    pip install -r requirements.txt