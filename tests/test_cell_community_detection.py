import unittest
import sys

from stereo.core.stereo_exp_data import AnnBasedStereoExpData
import anndata as ad
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

from stereo.algorithm.community_detection import CommunityDetection
from stereo.algorithm.ccd import *
from stereo.algorithm.ccd.utils import csv_to_anndata

class TestCellCommunityDetection(unittest.TestCase):

    def setUp(self, *args, **kwargs):
        super().setUp()

    def create_test_data(self, annotation_label: str):
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
        # create labels as strings ('0','1','2')
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

        # Organise x,y and labels (annotation) into one DataFrame
        df = pd.DataFrame(X_agg_sl, columns=['x', 'y'])
        df.loc[:, annotation_label] = y_agg_sl
        # Add cell_ID names as index values from DataFrame
        df['cell_ID'] = df.index

        return df

    def test_cell_community_detection_from_h5ad(self):
        annotation_label = 'annotation'
        # create synthetic data DataFrame
        df = self.create_test_data(annotation_label=annotation_label)

        """ Prepare anndata for CCD 
        This code can also be used if a .csv is available with spatial coordinates (x,y) and cell type annotation. 
        The .csv can be read using pandas function and then converted to AnnData format necessary for CCD.
        Coordinates are placed in .obsm['spatial'] ('X_spatial' and 'spatial_stereoseq' are also supported).
        Cell type annotation must be placed in .obs. Name od the annotation label is not defined, but needs to be 
        provided as an argument to CCD.
        """
        # gene expression is not needed, the X layer is filled with zeros
        adata = ad.AnnData(X=np.zeros(shape=(df.shape[0], 1), dtype=np.float32), dtype=np.float32)
        adata.obsm['spatial'] = df.loc[:, ['x', 'y']].values
        adata.obs[annotation_label] = df.annotation.values
        adata.obs_names = df.cell_ID

        adata.write_h5ad('synthetic_data.h5ad')
        stereoexp = AnnBasedStereoExpData('synthetic_data.h5ad')

        """Run CCD
        CCD can be run using the main class CommunityDetection. The object of this class requires only list of 
        slices (Anndata objects) and annotation label.
        If window size is not provided CCD calculates an optimal window size and sets sliding step to the half of 
        window size. Other default values for parameters can be found in README.
        """
        cd = CommunityDetection([stereoexp], annotation=annotation_label, plotting=2, hide_plots=True)
        cd.main()

    def test_cell_community_detection_from_csv(self):
        annotation_label = 'annotation'
        # create synthetic data DataFrame
        df = self.create_test_data(annotation_label=annotation_label)

        """ Prepare csv data for CCD 
        A DataFrame is created with cell_ID names, spatial coordinates (x,y) and cell type annotation, and saved as .csv.
        The .csv can be read using pandas function and then converted to AnnData format necessary for CCD
        using csv_to_anndata function. Name od the annotation label is not defined, but needs to be 
        provided as an argument to CCD. Using only the necessary data via .csv, instead of full .h5ad, the CCD
        takes less memory and time.
        """
        df.to_csv('synthetic_data.csv')
        # read csv, create Anndata object and use it to initialize the stereoexp object
        stereoexp = AnnBasedStereoExpData(h5ad_file_path=None, based_ann_data=csv_to_anndata('synthetic_data.csv', annotation=annotation_label))

        """ Run CCD
        CCD can be run using the main class CommunityDetection. The object of this class requires only list of 
        slices (Anndata objects) and annotation label.
        If window size is not provided CCD calculates an optimal window size and sets sliding step to the half of 
        window size. Other default values for parameters can be found in README.
        """
        cd = CommunityDetection([stereoexp], annotation=annotation_label)
        cd.main()


if __name__ == '__main__':
    unittest.main()
