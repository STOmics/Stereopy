Contributing Guide
~~~~~~~~~~~~~~~~~~

Contents
========
- `Contributing Code`_
- `Code Style`_
- `Project Structure`_
- `Testing`_

Contributing Code
-----------------
1. Clone the Stereopy source from GitHub::

    git clone https://github.com/BGIResearch/stereopy.git
2. Create a new branch for your PR and checkout to the new branch::

    git checkout -b new_branch_name
3. Add new functionality or fix bugs in your codebase.

4. After testing, update the relevant documentation, such as release notes, examples, etc.

5. Send `Pull Requests` to the dev branch. We will review your work and merge it with the main branch if there are no
performance or logical issues.

Code Style
----------
1. Coding requirements comply with pep8 specification.

2. The file name uses the snake case naming rule, while the class name uses the camel case naming rule::

    eg: file name: dim_reduce.py; class name: DimReduce

3. A variable should be used in a lenient snake case and should be as meaningful as possible, avoiding unintentional naming.

4. Comments should be perfect, and each file, function, and class should write its comments. We recommend using restructured Text as the docstring format for commenting on information.

5. Imports should be grouped in the following order::

    standard library imports
    related third party imports
    local application/library specific imports

You should put a blank line between each group of imports.

6. Functions and logic that are not implemented in the requirements but have been planned shall be marked with TODO. Confirm that a certain situation is faulty, and mark FIXME.

7. Use the logger in `log_manager` instead of your custom logger or print.

8. A new algorithm method should inherit with the base class `AlgorithmBase`, implementing your own `main` function in the child class.What's more, you can define the function name start with `test_`, and we will auto-test while we build new version.


Project Structure
-----------------
The Stereopy project:

- `stereo <stereo>`_: the root of the package.

  - `stereo/core <stereo/core>`_: the core code of stereo, which contains the base class and data structure of Stereopy.
  - `stereo/algorithm <stereo/algorithm>`_: the algorithm module, main analysis and implementation algorithms, which
    deals with methodology realization.
  - `stereo/image <stereo/image>`_: the image module, which deals with the tissue image related analysis, such as cell
    segmentation, etc.
  - `stereo/io <stereo/io>`_: the io module, which deals with the reading, writing and format conversion of different
    data structure, between our StereoExpData and AnnData, etc.
  - `stereo/plots <stereo/plots>`_: the plotting module, which contains all the plotting functions for visualization.
  - `stereo/utils <stereo/utils>`_: Some common processing scripts.
  - `stereo/tests <stereo/tests>`_: the Tests module, which contains all the test scripts.

Testing
-------
For each functional module, a corresponding test script should be created to ensure that the developed function is normal. All our test files are unified in the `tests <stereo/tests>`_ directory.
