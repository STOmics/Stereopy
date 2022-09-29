Cell correction 
===============
These examples show how to use Stereopy to correct cells.

Generally, there are two ways to do it.
  1. Correcting from bgef and mask.
  2. Correcting from gem and mask.

Note
--------
We have two versions of the algorithm, one is more slower but more accurate, another one is more faster but less accurate.
You can set the parameter `fast` to True to run the faster version, default to True.


Correcting from bgef and mask
------------------------------

On this way, you should specify the path of bgef by parameter `bgef_path`, the path of mask by parameter `mask_path` and the path of directory to save corrected result by parameter `out_dir`.

You can specify the count of processes by parameter `process_count`, default to 10.

Default to return an object of StereoExpData, if you set the parameter `only_save_result` to True, only return the path of cgef after correcting.

In the directory specified by out_dir, you can see some files, include:
  1. \*\*.raw.cellbin.gef - the cgef without correcting, generated from bgef and mask.
  2. \*\*.adjusted.gem - the gem after correcting.
  3. \*\*.adjusted.cellbin.gef - the cgef after correcting, generated from the \*\*.adjusted.gem.
  4. err.log - record the cells can not be corrected, these cells are not contained in \*\*.adusted.gem and \*\*.adjusted.cellbin.gef.

.. code:: python

    from stereo.tools import cell_correct

    bgef_path = "FP200000443TL_E2.bgef"
    mask_path = "FP200000443TL_E2_mask.tif"
    out_dir = "cell_correct_result"
    only_save_result = False
    fast = True
    data = cell_correct(out_dir=out_dir,
                        bgef_path=bgef_path,
                        mask_path=mask_path,
                        process_count=10,
                        only_save_result=only_save_result,
                        fast=fast)

Sometimes, you may not have corresponding mask but a ssdna image, you can generate mask from ssdna image.

Now, you should specify the path of ssdna image by parameter `image_path`, the path of model for predicting by parameter `model_path` and the type of model by parameter `model_type`.

The type of model only can be specified to deep-learning or deep-cell, more deails on `cell segmentation <https://stereopy.readthedocs.io/en/latest/Tutorials/cell_segmentation.html>`_.

You can also predict on gpu, specify gpu id by parameter `gpu`, if -1, predict on cpu.

In the `out_dir` directory, there is a new directory named deep-learning or deep-cell, it contains the generated mask whoes name ends with mask.tif.

.. code:: python

    from stereo.tools import cell_correct

    out_dir = "cell_correct_result"
    bgef_path = "SS200000561BL_B3.bgef"
    image_path = "SS200000561BL_B3_regist.tif"
    model_path = "cell_segmentation/seg_model_20211210.pth"
    model_type = "deep-learning"
    #model_path = "cell_segmentation_deepcell"
    #model_type = "deep-cell"
    gpu = -1
    only_save_result = False
    fast = True
    data = cell_correct(out_dir=out_dir,
                        bgef_path=bgef_path,
                        image_path=image_path,
                        model_path=model_path,
                        model_type=model_type,
                        gpu=gpu,
                        process_count=10,
                        only_save_result=only_save_result,
                        fast=fast)


Correcting from gem and mask
-----------------------------

On this way, you should specify the path of gem by parameter `gem_path`, the path of mask by parameter `mask_path` and the path of directory to save corrected result by parameter `out_dir`.

In the `out_dir` directory, you can also see a file named \*\*.bgef, this is the bgef generated from mask.

.. code:: python

    from stereo.tools import cell_correct

    gem_path = "FP200000443TL_E2.gem"
    mask_path = "FP200000443TL_E2_mask.tif"
    out_dir = "cell_correct_result"
    only_save_result = False
    fast = True
    data = cell_correct(out_dir=out_dir,
                      gem_path=gem_path,
                        mask_path=mask_path,
                        process_count=10,
                        only_save_result=only_save_result,
                        fast=fast)

Similar to the way on bgef and ssdna image, you can correct cells from gem and ssdna image.

.. code:: python

    from stereo.tools import cell_correct

    out_dir = "cell_correct_result"
    gem_path = "./SS200000561BL_B3.gem"
    image_path = "./SS200000561BL_B3_regist.tif"
    model_path = "./seg_model_20211210.pth"
    model_type = "deep-learning"
    #model_path = "./cell_segmentation_deepcell"
    #model_type = "deep-cell"
    gpu = -1
    only_save_result = False
    fast = True
    data = cell_correct(out_dir=out_dir,
                        gem_path=gem_path,
                        image_path=image_path,
                        model_path=model_path,
                        model_type=model_type,
                        gpu=gpu,
                        process_count=10,
                        only_save_result=only_save_result,
                        fast=fast)


Runing on jupyter notebook
---------------------------

Jupyter notebook can not support multiprocess directly, if you want to run on notebook, refer to the following two steps.

The first, you need to write the source code into a .py file by command %%writefile.

After running the example below, you should see a file named temp.py in current directory.

.. code:: python

    %%writefile temp.py
    from stereo.tools import cell_correct

    bgef_path = "FP200000443TL_E2.bgef"
    mask_path = "FP200000443TL_E2_mask.tif"
    out_dir = "cell_correct_result"
    only_save_result = False
    fast = True
    data = cell_correct(out_dir=out_dir,
                        bgef_path=bgef_path,
                        mask_path=mask_path,
                        process_count=10,
                        only_save_result=only_save_result,
                        fast=fast)

And the second, run the .py file by command %run

.. code:: python

    %run temp.py