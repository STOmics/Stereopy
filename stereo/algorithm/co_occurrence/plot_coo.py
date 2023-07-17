from natsort import natsorted
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from stereo.plots.plot_base import PlotBase


class PlotCoOccurrence(PlotBase):
    def co_occurrence_plot(self, groups=[], res_key = 'co_occurrence'):
        '''
        Visualize the co-occurence by line plot; each subplot represent a celltype, each line in subplot represent the 
        co-occurence value of the pairwise celltype as the distance range grow.
        :param cluster_res_key: The key of the cluster or annotation result of cells stored in data.tl.result which ought to be equal to cells in length.
        :param groups: Choose a few cluster to plot, plot all clusters by default.
        :param res_key: The key to store co-occurence result in data.tl.result
        :param savefig: The path to save plot
        :return: None
        '''
        data = self.stereo_exp_data
        if len(groups)==0:
            groups = natsorted(self.pipeline_res[res_key].keys())
        else:
            groups = natsorted(groups)
        nrow = int(np.sqrt(len(groups)))
        ncol = np.ceil(len(groups)/nrow).astype(int)
        # print(nrow, ncol)
        fig = plt.figure(figsize=(5*ncol, 5*nrow))
        axs = fig.subplots(nrow, ncol)
        # clust_unique = list(data.cells[cluster_res_key].astype('category').cat.categories)
        for i, g in enumerate(groups):
            interest = data.tl.result[res_key][g]
            if nrow == 1:
                if ncol == 1:
                    ax = axs
                else:
                    ax = axs[i]
            else:
                ax = axs[int(i/ncol)][(i%ncol)]
            ax.plot(interest, label = interest.columns)
            ax.set_title(g)
            ax.legend(fontsize = 7, ncol = max(1, nrow-1), loc='upper right')
        # if savefig != None:
        #     fig.savefig(savefig, dpi=300, bbox_inches='tight')
        # else:
        #     plt.show()
        return fig




    def co_occurrence_heatmap(self, cluster_res_key, dist_min=0, dist_max=10000, res_key='co_occurrence'):
        '''
        Visualize the co-occurence by heatmap; each subplot represent a certain distance, each heatmap in subplot represent the 
        co-occurence value of the pairwise celltype.
        :param data: An instance of StereoExpData, data.position & data.tl.result[use_col] will be used.
        :param use_col: The key of the cluster or annotation result of cells stored in data.tl.result which ought to be equal 
                        to cells in length.
        :param dist_min: Minimum distance interested, threshold between (dist_min,dist_max) will be plot
        :param dist_max: Maximum distance interested, threshold between (dist_min,dist_max) will be plot
        :param res_key: The key to store co-occurence result in data.tl.result
        :param savefig: The path to save plot
        :return: None
        '''
        from seaborn import heatmap
        for tmp in self.pipeline_res[res_key].values():
            break
        groups = [x for x in tmp.index if (x<dist_max)&(x>dist_min)]
        nrow = int(np.sqrt(len(groups)))
        ncol = np.ceil(len(groups)/nrow).astype(int)
        # print(nrow, ncol)
        fig = plt.figure(figsize=(9*ncol,8*nrow))
        axs = fig.subplots(nrow,ncol)
        for ax in axs:
            ax.set_axis_off()
        clust_unique = list(self.stereo_exp_data.cells[cluster_res_key].astype('category').cat.categories)
        for i, g in enumerate(groups):
            interest = pd.DataFrame({x: self.pipeline_res[res_key][x].T[g] for x in self.pipeline_res[res_key]})
            #interest = pd.DataFrame({x:data.tl.result[use_key][x].T[g] for x in clust_unique})
            if nrow == 1:
                if ncol == 1:
                    ax = axs
                else:
                    ax = axs[i]
            else:
                ax = axs[int(i/ncol)][(i%ncol)]
            if set(interest.index)==set(clust_unique) and set(interest.columns)==set(clust_unique):
                interest = interest.loc[clust_unique, clust_unique]
            heatmap(interest, ax=ax, center=0)
            ax.set_title('{:.4g}'.format(g))
            ax.set_axis_on()
            ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', va = 'top')
            #ax.legend(fontsize = 7, ncol = max(1, nrow-1), loc='upper right')
        # if savefig != None:
        #     fig.savefig(savefig, dpi=300, bbox_inches='tight')
        # else:
        #     plt.show()
        return fig