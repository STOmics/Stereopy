Contributing Guide
~~~~~~~~~~~~~~~~~~

Contents
========
- `Contributing Workflow`_
- `Creating Development Environment`_
- `Project Structure`_
- `Code Style`_
- `Testing`_
- `Algorithm Method Class`_


Contributing Workflow
---------------------
1. Fork the Stereopy repository to your own GitHub account, learn more in `GitHub Fork <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_.

2. Clone your fork of the repository, and create a new branch based on dev branch for your `Pull Requests`, create a virtual environment such as conda, see details at `Creating Development Environment`_.

3. Add new functionality or fix bugs in your codebase, don't forget to follow the rules in `Code Style`_.

4. After finishing an important work, we strongly recommend contributor to add `{your_work_tutorial}.ipynb`, whose style like `HotSpot Tutorial <https://stereopy.readthedocs.io/en/latest/Tutorials/hotspot.html>`_, into `stereopy/docs/source/Tutorials/`.

5. Run all tests, reading `Testing`_ for more details.

6. After testing, update the relevant documentation, such as release notes, examples, etc.

7. Open a `Pull Requests` to the dev branch. We will review your work and merge it with the main branch if there are no performance or logical issues.


Creating Development Environment
--------------------------------
1 Clone your repository.

.. code:: bash

    git clone https://github.com/{your_github_name}/stereopy.git

2 Checkout the dev branch, you can directly start your work at dev branch, or create a new branch.

.. code:: bash

    cd stereopy
    git checkout -b dev
    # Create a new branch for pulling requests
    git branch -c dev dev_my_pr

3 Install Stereopy for development.

.. code:: bash

    # Enter the source directory
    cd stereopy

    conda create -n stereopy_pr python==3.8

    # Install stereopy for developing
    python setup.py develop


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

Code Style
----------
1. Coding requirements comply with `PEP8 <https://legacy.python.org/dev/peps/pep-0008/#a-foolish-consistency-is-the-hobgoblin-of-little-minds>`_ specification.

2. The file name uses the snake case naming rule, while the class name uses the camel case naming rule, see `Algorithm Method Class`_.

3. A variable should be used in a lenient snake case and should be as meaningful as possible, avoiding unintentional naming.

4. Comments should be perfect. Each file, function and class should write its comments. We recommend using ReStructured Text as the docstring format for commenting on information.

.. code:: python

    # rst style example
    def example_func(data, debug=False, return_type='dict'):
        '''
        A useful func to do sth.

        :param data: the input data be transformed
        :param debug: default False, whether to show debug output
        :param return_type: default 'dict', return object type. Setting to 'dict', the result will be organized by python dictionary.
        '''
        pass

5. Imports should be grouped in the following order, You should put a blank line between each group of imports.

.. code:: python

    # standard library imports
    import time
    from copy import deepcopy

    # related third party imports
    import numpy as np

    # local application/library specific imports
    from ..log_manager import logger
    from .algorithm_base import AlgorithmBase, ErrorCode

6. Functions and logic that are not implemented in the requirements but have been planned shall be marked with TODO. Confirm that a certain situation is faulty, and mark FIXME.

7. Use the logger in `log_manager` instead of your custom logger or print.

8. A new algorithm method should inherit with the base class `AlgorithmBase` (see: `Algorithm Method Class`_).


Testing
-------
For each functional module, a corresponding test script should be created to ensure that the developed function is normal.

All our test files are unified in the `tests <stereo/tests>`_ directory.

.. code:: bash

    cd stereo/tests/
    pytest

Algorithm Method Class
----------------------
1 Add a new py file named `example_method` used in a lenient snake case. Within the file, a new algorithm method named by camel case.

2 And then, implement your own `main` function in the child class, you can also define the function name start with `test_`, and we will auto-test while we build new version.

.. code:: python

    # path: stereo/algorithm/example_method.py

    # standard library imports
    import time
    from copy import deepcopy

    # related third party imports
    import numpy as np

    # local application/library specific imports
    from ..log_manager import logger
    from .algorithm_base import AlgorithmBase, ErrorCode

    class Log1pFake(AlgorithmBase):

        def main(self, log_fast=True, inplace=True, verbose=False):
            """
                This is a fake log1p method.

                :param log_fast:
                :param inplace:
                :param verbose: TODO: verbose not finished
                :return:
            """

            not_used_variable = None
            ircorrect_spell_word = 'should be `incorrect`'
            the_very_beginning_time = time.time()

            if inplace:
                stereo_exp_data = self.stereo_exp_data
            else:
                stereo_exp_data = deepcopy(self.stereo_exp_data)

            if not log_fast:
                # FIXME: use time.sleep will stuck when this method is using in a web-api
                time.sleep(3.14159)
            stereo_exp_data.exp_matrix = np.log1p(stereo_exp_data.exp_matrix)

            if not inplace:
                self.pipeline_res['log1p'] = stereo_exp_data

            logger.info('log1p cost %.4f seconds', time.time() - the_very_beginning_time)
            return ErrorCode.Success

        def test_copy_safety(self):
            stereo_exp_data = deepcopy(self.stereo_exp_data)
            assert id(stereo_exp_data) != id(self.stereo_exp_data)
            assert id(stereo_exp_data.tl) != id(self.stereo_exp_data.tl)
            assert id(stereo_exp_data.plt) != id(self.stereo_exp_data.plt)
            assert id(stereo_exp_data.exp_matrix) != id(self.stereo_exp_data.exp_matrix)
