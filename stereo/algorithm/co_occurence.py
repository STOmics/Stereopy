# python core module
from collections import defaultdict
import time

# third part module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
import networkx as nx
from sklearn.metrics import pairwise_distances

# module in self project
import stereo as st # only used in test function to read data
from ..log_manager import logger
from .algorithm_base import AlgorithmBase, ErrorCode


class CoOccurence(AlgorithmBase):
    """
    docstring for CoOccurence
    :param 
    :return: 
    """
    def main(self, data, use_col, method='stereopy', dist_thres = 300, steps = 10, genelist = None, gene_thresh = 0):
        """
        Co-occurence calculate the score or probability of a particular celltype or cluster of cells is co-occurence with 
          another in spatial.  
        We provided two method for co-occurence, 'squidpy' for method in squidpy, 'stereopy' for method in stereopy
        :param data: An instance of StereoExpData, data.position & data.tl.result[use_col] will be used.
        :param use_col: The key of the cluster or annotation result of cells stored in data.tl.result which ought to be equal 
                        to cells in length.
        :param method: The metrics to calculate co-occurence choose from ['stereopy', 'squidpy'], 'squidpy' by default.
        :param dist_thres: The max distance to measure co-occurence. Only used when method=='stereopy'
        :param steps: The steps to generate threshold to measure co-occurence, use along with dist_thres, i.e. default params 
                      will generate [30,60,90......,270,300] as threshold. Only used when method=='stereopy'
        :param genelist: Calculate co-occurence between use_col & genelist if provided, otherwise calculate between clusters 
                         in use_col. Only used when method=='stereopy'
        :param gene_thresh: Threshold to determine whether a cell express the gene. Only used when method=='stereopy'
        :return: the input data with co_occurrence result in data.tl.result['co-occur']
        """
        if method == 'stereopy':
            self.co_occurrence(data, use_col, dist_thres = dist_thres, steps = steps, genelist = genelist, gene_thresh = gene_thresh)
        elif method == 'squidpy':
            self.co_occurrence_squidpy(data, use_col)
        return data


    def co_occurrence_squidpy(self, data, use_col):
        """
        Squidpy mode to calculate co-occurence, result same as squidpy
        :param data: An instance of StereoExpData, data.position & data.tl.result[use_col] will be used.
        :param use_col: The key of the cluster or annotation result of cells stored in data.tl.result which ought to be equal 
                        to cells in length.
        :return: co_occurrence result, also written in data.tl.result['co-occur']
        """
        dist_ori = pairwise_distances(data.position, data.position, metric='euclidean')
        #thresh_min, thresh_max = np.min(dist_ori), np.max(dist_ori)
        thresh_min, thresh_max = self._find_min_max(data.position)
        thresh = np.linspace(thresh_min, thresh_max, num=50)    
        clust_unique = list(data.tl.result[use_col]['group'].astype('category').cat.categories)
        clust = list(data.tl.result[use_col]['group'].astype('category').cat.codes)
        num = len(clust_unique)
        out = np.zeros((num, num, thresh.shape[0] - 1))
        for ep in range(thresh.shape[0] - 1):
            co_occur = np.zeros((num, num))
            probs_con = np.zeros((num, num))
            thresh_l, thresh_r = thresh[ep], thresh[ep+1]
            idx_x, idx_y = np.nonzero((dist_ori <= thresh_r) & (dist_ori > thresh_l))
            x = data.tl.result[use_col]['group'].astype('category').cat.codes[idx_x]
            y = data.tl.result[use_col]['group'].astype('category').cat.codes[idx_y]
            for i, j in zip(x, y):
                co_occur[i, j] += 1
            probs_matrix = co_occur / np.sum(co_occur)
            probs = np.sum(probs_matrix, axis=1)
        
            for c,d in enumerate(clust_unique):
                probs_conditional = co_occur[c] / np.sum(co_occur[c])
                probs_con[c, :] = probs_conditional / probs
        
            out[:, :, ep] = probs_con
        ret = {}
        for i, j in enumerate(clust_unique):
            tmp = pd.DataFrame(out[i]).T
            tmp.columns = clust_unique
            tmp.index = thresh[1:]
            ret[j] = tmp
        data.tl.result['co-occur'] = ret
        return ret
    
    def _find_min_max(self, spatial):
        '''
        Helper to calculate distance threshold in squidpy mode
        param: spatial: the cell position of data
        return: thres_min, thres_max for minimum & maximum of threshold
        '''
        coord_sum = np.sum(spatial, axis=1)
        min_idx, min_idx2 = np.argpartition(coord_sum, 2)[:2]
        max_idx = np.argmax(coord_sum)
        # fmt: off
        thres_max = pairwise_distances(spatial[min_idx, :].reshape(1, -1), spatial[max_idx, :].reshape(1, -1))[0, 0] / 2.0
        thres_min = pairwise_distances(spatial[min_idx, :].reshape(1, -1), spatial[min_idx2, :].reshape(1, -1))[0, 0]
        # fmt: on
        return thres_min, thres_max
    
    def co_occurrence(self, data, use_col, dist_thres = 300, steps = 10, genelist = None, gene_thresh = 0):
        '''
        Stereopy mode to calculate co-occurence, the score of result['A']['B'] represent the probablity of 'B' occurence around 
          'A' in distance of threshold
        :param data: An instance of StereoExpData, data.position & data.tl.result[use_col] will be used.
        :param use_col: The key of the cluster or annotation result of cells stored in data.tl.result which ought to be equal 
                        to cells in length.
        :param method: The metrics to calculate co-occurence choose from ['stereopy', 'squidpy'], 'squidpy' by default.
        :param dist_thres: The max distance to measure co-occurence. 
        :param steps: The steps to generate threshold to measure co-occurence, use along with dist_thres, i.e. default params 
                      will generate [30,60,90......,270,300] as threshold. 
        :param genelist: Calculate co-occurence between use_col & genelist if provided, otherwise calculate between clusters 
                         in use_col. 
        :param gene_thresh: Threshold to determine whether a cell express the gene. 
        :return: co_occurrence result, also written in data.tl.result['co-occur']
        '''
        #from collections import defaultdict
        #from scipy import sparse
        from sklearn.metrics import pairwise_distances
        dist_ori = pairwise_distances(data.position, data.position, metric='euclidean')
        if isinstance(genelist, np.ndarray):
            genelist = list(genelist)
        elif isinstance(genelist, list):
            genelist = genelist
        elif isinstance(genelist, str):
            genelist = [genelist]
        elif isinstance(genelist, int):
            genelist = [genelist]
            
        thresh = np.linspace(0, dist_thres, num=steps+1)   
        out = {}
        for ep in range(thresh.shape[0] - 1):
            thresh_l, thresh_r = thresh[ep], thresh[ep+1]
            dist = dist_ori.copy()
            dist[(dist>=thresh_l)&(dist<thresh_r)]=-1
            dist[dist>-1]=0
            dist[dist==-1]=1
            if genelist is None:
                #df = data.obs[['Centroid_X', 'Centroid_Y', use_col]]
                count = {x:0 for x in data.tl.result[use_col]['group'].unique()}
                ret = defaultdict(dict)
                for x in data.tl.result[use_col]['group']:
                    for y in data.tl.result[use_col]['group']:
                        ret[x][y]=0
                for x, y in enumerate(data.tl.result[use_col]['group']):
                    for z in np.unique(data.tl.result[use_col]['group'].to_numpy()[dist[x].astype(bool)]):
                        ret[y][z]+=1
                    count[y]+=1
                ret=pd.DataFrame(ret)
                ret=ret/count
                out[thresh_r] = ret
            else:
                ret = defaultdict(dict)
                for x in data.tl.result[use_col]['group']:
                    for y in genelist:
                        ret[x][y]=0
                count = {x:0 for x in data.tl.result[use_col]['group'].unique()}
                gene_exp_dic ={}
                for z in genelist:
                    if sparse.issparse(data.exp_matrix):
                        gene_exp=data.exp_matrix[:, np.where(data.genes.gene_name==z)].copy().todense().flatten()
                    else:
                        gene_exp=data.exp_matrix[:, np.where(data.genes.gene_name==z)].copy().flatten()
                    gene_exp[gene_exp<gene_thresh] = 0
                    gene_exp_dic[z] = gene_exp
                for x, y in enumerate(data.tl.result[use_col]['group']):
                    for z in genelist:
                        if (gene_exp_dic[z]*dist[x]).sum()>0:
                            ret[y][z] += 1
                    count[y]+=1
                ret=pd.DataFrame(ret)
                ret=ret/count
                out[thresh_r] = ret
        ret = {}
        for x in out[thresh_r].index:
            tmp = {}
            for ep in out:
                tmp[ep] = out[ep].T[x]
            ret[x] = pd.DataFrame(tmp).T
        data.uns['co-occur'] = ret
        return ret   



    def co_occurrence_plot(self, data, use_col, groups=[], use_key = 'co-occur', savefig=None):
        '''
        Visualize the co-occurence by line plot; each subplot represent a celltype, each line in subplot represent the 
        co-occurence value of the pairwise celltype as the distance range grow.
        :param data: An instance of StereoExpData, data.position & data.tl.result[use_col] will be used.
        :param use_col: The key of the cluster or annotation result of cells stored in data.tl.result which ought to be equal 
                        to cells in length.
        :param groups: Choose a few cluster to plot, plot all clusters by default.
        :param use_key: The key to store co-occurence result in data.tl.result
        :param savefig: The path to save plot
        :return: None
        '''
        if len(groups)==0:
            groups = data.tl.result[use_key].keys()
        nrow = int(np.sqrt(len(groups)))
        ncol = np.ceil(len(groups)/nrow).astype(int)
        print(nrow, ncol)
        fig = plt.figure(figsize=(5*ncol,5*nrow))
        axs = fig.subplots(nrow,ncol)
        clust_unique = list(data.tl.result[use_col]['group'].astype('category').cat.categories)
        for i, g in enumerate(groups):
            interest = data.tl.result[use_key][g]
            if nrow == 1:
                if ncol == 1:
                    ax = axs
                else:
                    ax = axs[i]
            else:
                ax = axs[int(i/ncol)][(i%ncol)]
            ax.plot(interest, label = interest.columns )
            ax.set_title(g)
            ax.legend(fontsize = 7, ncol = max(1, nrow-1), loc='upper right')
        if savefig != None:
            fig.savefig(savefig, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    

    
    
    def co_occurrence_heatmap(self, data, use_col, dist_min=0, dist_max=10000, use_key = 'co-occur', savefig=None):
        '''
        Visualize the co-occurence by heatmap; each subplot represent a certain distance, each heatmap in subplot represent the 
        co-occurence value of the pairwise celltype.
        :param data: An instance of StereoExpData, data.position & data.tl.result[use_col] will be used.
        :param use_col: The key of the cluster or annotation result of cells stored in data.tl.result which ought to be equal 
                        to cells in length.
        :param dist_min: Minimum distance interested, threshold between (dist_min,dist_max) will be plot
        :param dist_max: Maximum distance interested, threshold between (dist_min,dist_max) will be plot
        :param use_key: The key to store co-occurence result in data.tl.result
        :param savefig: The path to save plot
        :return: None
        '''
        from seaborn import heatmap
        for tmp in data.tl.result[use_key].values():
            break
        groups = [x for x in tmp.index if (x<dist_max)&(x>dist_min)]
        nrow = int(np.sqrt(len(groups)))
        ncol = np.ceil(len(groups)/nrow).astype(int)
        print(nrow, ncol)
        fig = plt.figure(figsize=(9*ncol,8*nrow))
        axs = fig.subplots(nrow,ncol)
        clust_unique = list(data.tl.result[use_col]['group'].astype('category').cat.categories)
        for i, g in enumerate(groups):
            interest = pd.DataFrame({x:data.tl.result[use_key][x].T[g] for x in data.tl.result[use_key]})
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
            heatmap(interest, ax=ax)
            ax.set_title('{:.4g}'.format(g))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', va = 'top')
            #ax.legend(fontsize = 7, ncol = max(1, nrow-1), loc='upper right')
        if savefig != None:
            fig.savefig(savefig, dpi=300, bbox_inches='tight')
        else:
            plt.show()



    def test_co_occurence(self):
        '''
        test fuction to chech codes 
        '''
        #mouse_data_path = 'data/SS200000135TL_D1.cellbin.gef'
        #data = st.io.read_gef(file_path=mouse_data_path, bin_type='cell_bins') 
        mouse_data_path = '/jdfssz2/ST_BIOINTEL/P20Z10200N0039/06.groups/04.Algorithm_tools/caolei2/Stereopy/data/SS200000135TL_D1.cellbin.h5ad'
        data = st.io.read_stereo_h5ad(file_path=mouse_data_path, use_raw=False, use_result=True) 
        self.co_occurrence(data, use_col='leiden')
        self.co_occurrence_plot(data, use_col='leiden',groups=['1','2','3','4', '5'], savefig = './co_occurrence_plot.png')
        self.co_occurrence_heatmap(data, use_col='leiden', dist_max=80, savefig = './co_occurrence_plot.png')
        return data

if __name__ == '__main__':
    test = CoOccurence()
    data = test.test_co_occurence()
