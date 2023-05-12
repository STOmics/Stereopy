Contributing
==============


Project Structure
-----------------

- `stereo <https://github.com/BGIResearch/stereopy/tree/main/stereo>`_: the root of the package.
- `stereo/core <https://github.com/BGIResearch/stereopy/tree/main/stereo/core>`_: the core code of stereo which contains the base classes and data structure of Stereopy.
- `stereo/algorithm <https://github.com/BGIResearch/stereopy/tree/main/stereo/algorithm>`_: the algorithm module, containing main analysis and implementation algorithms which deals with methodology realization.
- `stereo/image <https://github.com/BGIResearch/stereopy/tree/main/stereo/image>`_: the image module which deals with analysis related to the image file, such as cell segmentation, etc.
- `stereo/io <https://github.com/BGIResearch/stereopy/tree/main/stereo/io>`_: the io module which deals with reading, writing and format conversion of different data structures, between StereoExpData and AnnData, etc.
- `stereo/plots <https://github.com/BGIResearch/stereopy/tree/main/stereo/plots>`_: the plotting module which contains all plotting functions for visualization.
- `stereo/utils <https://github.com/BGIResearch/stereopy/tree/main/stereo/utils>`_: the common processing scripts.
- `stereo/tests <https://github.com/BGIResearch/stereopy/tree/main/tests>`_: the test module which contains all test scripts.


Contributing Guide
---------------------
1. **Fork** the Stereopy repository to your own GitHub account, learn more in `GitHub Fork <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_.

2. **Clone** your fork of the repository, create a new branch based on dev branch for your `Pull Requests`, and create a virtual environment to install Stereopy of Development Version.

3. **Add new functions** or **fix bugs** in your codebase, don't forget to follow the rules of `Code Style`_.

4. After completing **code work**, we strongly recommend contributors to add `{your_work_tutorial}.ipynb`, whose style like `Spatial hotSpot tutorial <https://stereopy.readthedocs.io/en/latest/Tutorials/hotspot.html>`_, into `stereopy/docs/source/Tutorials/`.

5. **Run all tests**, and read `Test`_ for more details.

6. After testing, update relevant **documentation**, such as release notes, examples, etc.

7. `Pull Requests` to dev branch, and we will review your work and merge it into the main branch if there are no issues of performance or logic.


Development Environment
--------------------------------

Clone your repository.

.. code:: bash

    git clone https://github.com/{your_github_name}/stereopy.git


Check out the dev branch, you can directly start work at dev branch, or create a new branch.

.. code:: bash

    cd stereopy
    git checkout -b dev
    # Create a new branch for pulling requests
    git branch -c dev dev_my_pr


Install Stereopy of development version.

.. code:: bash

    # Enter the source directory
    cd stereopy

    conda create -n stereopy_pr python==3.8

    # Install stereopy for developing
    python setup.py develop


Code Style
----------
1. Coding requirements comply with `PEP8 <https://legacy.python.org/dev/peps/pep-0008/#a-foolish-consistency-is-the-hobgoblin-of-little-minds>`_ specification.

2. The file name should use the snake case naming rule, while the class name should use the camel case naming rule, see `Algorithm Method Class`_.

3. A variable should be used in a lenient snake case and should be as meaningful as possible, avoiding unintentional naming.

4. Comments, given to each file, function and class, should be as complete and detailed as possible. We recommend using `ReStructured Text` as the docstring format for marks.

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

5. Imports should be grouped in the following order and a blank line should be put between each pair of imports.

.. code:: python

    # standard library imports
    import time
    from copy import deepcopy

    # related third party imports
    import numpy as np

    # local application/library specific imports
    from ..log_manager import logger
    from .algorithm_base import AlgorithmBase, ErrorCode

6. Functions and logic, which are not implemented in the requirements but have been planned, should be marked with `TODO`. Confirm that a certain situation is faulty, and mark `FIXME`.

7. Use the logger in `log_manager` instead of your custom logger or print.

8. A new algorithm method should inherit with the base class `AlgorithmBase` (see: `Algorithm Method Class`_).


Test
-----
For each function module, a corresponding test script should be created to ensure that the developed function is normal.

All test files are unified in the `tests <https://github.com/BGIResearch/stereopy/tree/main/tests>`_ directory.

.. code:: bash

    cd stereo/tests/
    pytest


Algorithm Method Class
----------------------
1. Add a new py file named `example_method` using snake-case naming. In the file, use camel-case naming for the algorithm method.

2. Then implement your own `main` function in the child class, you can also define the function name start with `test_`, and we will auto-test while we build new version.

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
