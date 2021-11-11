Contributing guide
~~~~~~~~~~~~~~~~~~

Contents
========
- `Contributing code`_
- `Code style`_
- `Project structure`_
- `Writing documentation`_
- `Making a release`_
- `Testing`_

Contributing code
-----------------
1. Clone the Stereopy source from github::

    git clone https://github.com/BGIResearch/stereopy.git
2. Create a new branch for your PR and checkout to the new branch::

    git checkout -b new_branch_name

3. Add the new feature or fit the bugs in your codebase.
4. After all tests passing, update the related documentation, such as release note, example and so on.
5. Make a Pull Requests back to the dev branch, We will review the submitted code, and the merge to the main branch after there is no problem.

Code style
----------
1. Coding requirements comply with pep8 specification.
2. The file naming adopts lowercase and underscore uniformly; the name of the class adopts the camel case naming method.

    eg: file name: dim_reduce.py; class name: DimReduce

3. Variable naming should be in lower case and should be as meaningful as possible, avoiding unintentional naming.
4. The comment information should be perfect, and each file, function, and class should write its comment information.
   We recommend using restructured Text as the docstring format of comment information.

Project structure
-----------------
The stereopy project:

- `stereo <stereo>`_: the root of the package.

  - `stereo/core <stereo/core>`_: the core code of stereo, which contains the base class and data structure of stereopy.
  - `stereo/algorithm <stereo/algorithm>`_: the algorithm module, main analysis and implementation algorithms, which
    deals with methodology realization.
  - `stereo/image <stereo/image>`_: the image module, which deals with the tissue image related analysis, such as cell
    segmentation, etc.
  - `stereo/io <stereo/io>`_: the io module, which deals with the reading, writing and format conversion of different
    data structure, between our StereoExpData and AnnData, etc.
  - `stereo/plots <stereo/plots>`_: the plotting module, which contains all the plotting functions for visualization.
  - `stereo/utils <stereo/utils>`_: Some common processing scripts.
  - `stereo/tests <stereo/tests>`_: the Tests module, which contains all the test scripts.

Writing documentation
---------------------
We use Sphinx to auto-generate documentation in multiple formats. Sphinx is built of reStructured text and, when using
sphinx most of what you type is reStructured text.

Installing Sphinx::

    pip install sphinx

    pio install sphinx_rtd_theme

    pip install recommonmark   # surpport md

Make a doc directory, and start sphinx::

    cd stereo

    mkdir docs

    # start sphinx
    sphinx-quickstart

Modify the config file:
    In doc/source directory is now a python file called conf.py. This is the file that controls the basics of how 
    sphinx runs when you run a build. For more information about the file, see the official 
    `Sphinx <https://www.sphinx-doc.org/en/master/usage/configuration.html>`_ document.

Write the Tutorials/example what you need to update:
    Tutorials and examples are hosted on a separate repository called 
    `Tutorials <https://github.com/BGIResearch/stereopy/tree/main/docs/source/Tutorials>`_.

Make and build html::

    make clean

    make html

    open index.html

docstrings format
`````````````````
We use the reStructured text style for writing docstrings. If you’re unfamiliar with the reStructuredText (rst) format,
see the `link <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_.

Making a release
----------------
This part is to tell developers how to publish stereopy to PyPi.

Checking the environment::

    # First, install twine
    pip install twine

    # make a build
    python setup.py sdist bdist_wheel

    # check the build
    twine check dist/*

Making release::

    # Tag the version info
    git tag {version}

    # Build distributions and wheel
    python setup.py sdist bdist_wheel

    # Check whether the compilation result can be installed successfully
    # eg: pip install dist/stereopy-{version}-py3-none-any.whl

    # push the tag to github
    git push origin {version}

    # Upload wheel and code distribution to PyPi
    twine upload dist/*

After any release has been made, create a new release notes file for the next feature and bugfix release.

Testing
-------
For each functional module, a corresponding test script should be created to ensure that the developed function
is normal. All our test files are unified in the `tests <stereo/tests>`_ directory.
