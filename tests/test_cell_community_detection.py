import unittest
import sys

from stereo.core.stereo_exp_data import AnnBasedStereoExpData
import anndata as ad
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

from community_detection import CommunityDetection
from ccd import *

class TestCellCommunityDetection(unittest.TestCase):

    def setUp(self, *args, **kwargs):
        super().setUp()


    def test_cell_community_detection(self):
        """ Create synthetic data
        Synthetic data is created using the scikit learn datasets make_blobs function.
        Three blobs are arranged to provide a simple case for community detection:
        one cell type spread throughout the tissue (magenta), and two have similar shapes that
        partially overlap (green and gold).
        The distribution of the spots is not uniform, but represents a good example for CCD.
        """
        n_samples = 10000
        random_state = 170
        # create basic blob (magenta)
        X, y = make_blobs(n_samples=n_samples, centers=1, center_box=(0, 10), cluster_std=[1],
                          random_state=random_state)
        # exponential transform of the blob (green)
        X_sq = np.array([[i, i * i * i + 2 * np.random.random_sample()] for i, j in X])
        # translation of exponentially transformed blob to create the third blob (gold)
        X_sq2 = X_sq + (0, 1)

        # merge data
        X_agg = np.vstack((X, X_sq, X_sq2))
        # adjustment of coordinates to a range that is most common in the ST datasets
        X_agg[:, 0] = (X_agg[:, 0] * 2000).astype(int)
        X_agg[:, 1] = (X_agg[:, 1] * 1000).astype(int)
        # create labels (0,1,2)
        y_agg = np.array([0] * len(X) + [1] * len(X_sq) + [2] * len(X_sq2))
        # define color for each spot based on labels (0 - magenta, 1 - green, 2 - gold)
        y_color = np.array(["#be398d"] * len(X) + ["#4fb06d"] * len(X_sq) + ["#d49137"] * len(X_sq2))

        # slice the data to fit the frame [0:2800, 0:4000]
        slice_mask = (X_agg[:, 0] > 0) & (X_agg[:, 0] < 2800) & (X_agg[:, 1] > 0) & (X_agg[:, 1] < 4000)
        X_agg_sl = X_agg[slice_mask]
        y_agg_sl = y_agg[slice_mask]
        y_color_sl = y_color[slice_mask]

        # display the generated sample
        plt.figure
        plt.scatter(X_agg_sl[:, 0], X_agg_sl[:, 1], c=y_color_sl, s=0.25)
        plt.title("Synthetic sample with 3 cell types")
        plt.show()

        """### Prepare anndata for CCD 
        This code can also be used if a .csv is available with spatial coordinates (x,y) and cell type annotation. 
        The .csv can be read using pandas function and then converted to AnnData format necessary for CCD.
        Coordinates are placed in .obsm['spatial'] ('X_spatial' and 'spatial_stereoseq' are also supported).
        Cell type annotation must be placed in .obs. Name od the annotation label is not defined, but needs to be 
        provided as an argument to CCD.
        It is also necessary to provide .uns['annotation_colors'], a list of colors for each cell type.
        """
        # Organise x,y and labels (annotation) into one DataFrame
        df = pd.DataFrame(X_agg_sl, columns=['x', 'y'])
        df.loc[:, 'annotation'] = y_agg_sl

        # TODO: Ugly: Replace.
        # gene expression is not needed, the X layer is filled with zeros
        adata = ad.AnnData(X=np.zeros(shape=(df.shape[0], 1)), dtype=np.float32)
        # the range of synthetic data
        adata.obsm['spatial'] = df.loc[:, ['x', 'y']].values
        adata.obs['annotation'] = df.annotation.values.astype('str')
        adata.uns['annotation_colors'] = ["#be398d", "#4fb06d", "#d49137"]

        adata.write_h5ad('ugly.h5ad')
        stereoexp = AnnBasedStereoExpData('ugly.h5ad')


        """Run CCD
        CCD can be run using the main class CommunityDetection. The object of this class requires only list of 
        slices (Anndata objects) and annotation label.
        If window size is not provided CCD calculates an optimal window size and sets sliding step to the half of 
        window size. Other default values for parameters can be found in README.
        Plotting argument can be changed to provide different levels of data and results visualization. 
        Its values goes from 0 to 5 (default 2).
        """
        cd = CommunityDetection([stereoexp], annotation='annotation')
        cd.main()


if __name__ == '__main__':
    unittest.main()
