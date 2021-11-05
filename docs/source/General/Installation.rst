Installation
============

Pre-built installation (recommended)
------------------------------------

.. code:: bash

    pip install stereopy

Source installation
--------------------------------------------

To install stereopy from source, you need:

- HDF5 1.8.4 or newer with development headers
- A C compiler

On Unix platforms, you also need pkg-config unless you explicitly specify a path for HDF5 as described below.

You can specify build options for stereopy as environment variables when you build it from source:

.. code:: bash

    git clone https://github.com/BGIResearch/stereopy.git
    cd stereopy
    HDF5_DIR=/path/to/hdf5 python setup.py install

The supported build options are:

 - To specify where to find HDF5, use one of these options:
    - HDF5_LIBDIR and HDF5_INCLUDEDIR: the directory containing the compiled HDF5 libraries and the directory containing the C header files, respectively.
    - HDF5_DIR: a shortcut for common installations, a directory with lib and include subdirectories containing compiled libraries and C headers.
    - HDF5_PKGCONFIG_NAME: A name to query pkg-config for. If none of these options are specified, stereopy will query pkg-config by default for hdf5, or hdf5-openmpi if building with MPI support.

Source installation on OSX/MacOS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HDF5 and Python are most likely in your package manager (e.g. Homebrew, Macports, or Fink). Be sure to install the development headers, as sometimes they are not included in the main package.

XCode comes with a C compiler (clang), and your package manager will likely have other C compilers for you to install.

Source installation on Linux/Other Unix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HDF5 and Python are most likely in your package manager. A C compiler almost definitely is, usually there is some kind of metapackage to install the default build tools, e.g. build-essential, which should be sufficient for our needs. Make sure that that you have the development headers, as they are usually not installed by default. They can usually be found in python-dev or similar and libhdf5-dev or similar.

Source installation on Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Installing from source on Windows is a much more difficult prospect than installing from source on other OSs, as not only are you likely to need to compile HDF5 from source, everything must be built with the correct version of Visual Studio. Additional patches are also needed to HDF5 to get HDF5 and Python to work together.
