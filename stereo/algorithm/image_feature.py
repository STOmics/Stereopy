# python core module
from collections import defaultdict
import time

# third part module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import sparse
import cv2

# module in self project
import stereo as st # only used in test function to read data
from ..log_manager import logger
from .algorithm_base import AlgorithmBase, ErrorCode

class ImageFeature(AlgorithmBase):
    """
    docstring for CoOccurence
    :param 
    :return: 
    """
    def main(self, im, density_thred = 30, ori_im=None):
        """
        read a binary image im, use connectedComponentsWithStats & findContours 
          to get all kinds of image feature 
        :param im: Image array of segmented cell mask, generate from the following image
                   ## im=Image.open('./data/thymus/SS200000150TL_E5_mask.tif')
                   ## im = np.array(im)
        :param density_thred: the length of square edge in density_calculator * 0.5
        :param ori_im: Image array of original regist image
        :return: a pd.DataFrame with shape of cells * features(19)
        """
        from scipy import spatial
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(im, connectivity=8)
        contours = cv2.findContours(im,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cnt = {'labels_id':[], 'perimeter':[], 'max_len':[], 'direct':[]}
        for x in contours[0]:
            tmp = []
            for y in x:
                tmp.append(labels[y[0][1], y[0][0]])
            cnt['labels_id'].append(max(tmp,key=tmp.count))
            cnt['perimeter'].append(len(x))
            arr = x[:,0,:]
            dist_mat = spatial.distance_matrix(arr,arr)
            cnt['max_len'].append(np.max(dist_mat))
            t = arr[int(np.argmax(dist_mat)/len(arr))]- arr[np.argmax(dist_mat)%len(arr)]
            if t[0] == 0:
                cnt['direct'].append(np.pi/2)
            else:
                cnt['direct'].append(np.arctan(t[1]/t[0]))
        cnt = pd.DataFrame(cnt)
        stats = pd.DataFrame(stats)
        stats.columns = ['x','y','height','width', 'area']
        stats = stats.drop(0, axis=0)
        stats['Centroid_X'] = centroids[1:, 1]
        stats['Centroid_Y'] = centroids[1:, 0]
        d1,d2,d3,d4,d5,d6,d7,d8,d9,perimeter,max_len,direct = [],[],[],[],[],[],[],[],[],[],[],[]
        tik = time.time()
        count = 0
        for row in stats.iterrows():
            count+=1
            if count%10000==0:
                print(time.time()-tik)
                tik = time.time()
            d1.append(self._density_calculator(stats, row, density_thred,-3,-1,3,1))
            d2.append(self._density_calculator(stats, row, density_thred,-1,1,3,1))
            d3.append(self._density_calculator(stats, row, density_thred,1,3,3,1))
            d4.append(self._density_calculator(stats, row, density_thred,-3,-1,1,-1))
            d5.append(self._density_calculator(stats, row, density_thred,-1,1,1,-1))
            d6.append(self._density_calculator(stats, row, density_thred,1,3,1,-1))
            d7.append(self._density_calculator(stats, row, density_thred,-3,-1,-1,-3))
            d8.append(self._density_calculator(stats, row, density_thred,-1,1,-1,-3))
            d9.append(self._density_calculator(stats, row, density_thred,1,3,-1,-3))
            tmp = cnt.loc[cnt['labels_id']==row[0]]
            if len(tmp)==0:
                perimeter.append(0)
                max_len.append(0)
                direct.append(0)
            elif len(tmp)==1:
                perimeter.append(list(tmp['perimeter'])[0])
                max_len.append(list(tmp['max_len'])[0])
                direct.append(list(tmp['direct'])[0])
            else:
                tmp = tmp.sort_values('perimeter')
                perimeter.append(list(tmp['perimeter'])[-1])
                max_len.append(list(tmp['max_len'])[-1])
                direct.append(list(tmp['direct'])[-1])
        stats['density1'] = d1
        stats['density2'] = d2
        stats['density3'] = d3
        stats['density4'] = d4
        stats['density5'] = d5
        stats['density6'] = d6
        stats['density7'] = d7
        stats['density8'] = d8
        stats['density9'] = d9
        stats['perimeter'] = perimeter
        stats['max_len'] = max_len
        stats['direct'] = direct
        stats = stats[['height','width', 'area', 'Centroid_X', 'Centroid_Y',\
                       'perimeter', 'max_len', 'direct', \
                       'density1', 'density2', 'density3',\
                       'density4', 'density5', 'density6',\
                       'density7', 'density8', 'density9']]
        stats['near_circularity'] = (stats['area']*4*np.pi) / (stats['perimeter'])**2
        if ori_im != None:
            stats = self._mean_gray_calculator(stats, labels, ori_im)
        #feature_ann = anndata.AnnData(stats)
        #feature_ann.obsm['spatial'] = stats[['Centroid_Y', 'Centroid_X']].to_numpy()
        return stats #, feature_ann

    def _density_calculator(self, stats, row, density_thred, l,r,u,d):
        """
        helper to calculator cell density, this is the main rate-determining step, I'm trying to make it better
        :param stats: contain X, Y feature to calculate density for each cell
        :param row: select cell row to calculate
        :param density_thred: the length of square edge in density_calculator * 0.5
        :param l,r,u,d: left, right, up, down boundary edge weight; (-1,1,1,-1) means to calculate density based on the row as center.
        :return: An intger represent the cell counts of the specific region
        """
        left = row[1]['Centroid_X']+l*density_thred
        right = row[1]['Centroid_X']+r*density_thred
        up = row[1]['Centroid_Y']+u*density_thred
        down = row[1]['Centroid_Y']+d*density_thred
        #print(left, right, up, down)
        return len(stats.loc[(stats['Centroid_X']<right)&(left<stats['Centroid_X'])&(stats['Centroid_Y']<up)&(down<stats['Centroid_Y'])])

    def _mean_gray_calculator(self, stats, labels, ori_im):
        """
        helper to calculate mean gray scale from the original image
        :param stats: container for result to write in
        :param labels: cell label image of segemented cell mask, generate from cv2.connectedComponentsWithStats
        :param ori_im: Image array of original regist image
        :return: stats with the mean_gray feature
        """
        ret = {}
        from scipy import sparse
        sp_label = sparse.coo_array(labels)
        df = pd.DataFrame({'values':sp_label.data, 'x':sp_label.row, "y":sp_label.col})
        for x, y in df.groupby('values'):
            ret[x] = np.mean(ori_im[y['x'], y['y']])
        stats['mean_gray'] = [ret[x] for x in stats.index()]
        return stats


    def mapping_labels(self, data, labels, feature, reverse_xy=True):
        """
        map the cell ID in labels with data.position by the center position
        param: data: if adata provided, the map result will be added to adata.obs
        param: labels: cell label image of segemented cell mask, generate from following codes :
               ## im=Image.open('./data/thymus/SS200000150TL_E5_mask.tif')
               ## im = np.array(im)
               ## retval, labels, stats, centroids = cv2.connectedComponentsWithStats(im, connectivity=8)
        param: feature: Image feature dataframe
        param: reverse_xy: bool value to determine whether to reverse the image by xy
        :return: the dict of cellname -> cellID in labels
        """
        tmp = data.position.astype(int)
        if reverse_xy :
            tmp = tmp[:, [1,0]]
        cl = labels[tmp[:,0], tmp[:,1]]
        cellid2loc = dict(zip(list(data.cells.cell_name), list(cl)))
        data.tl.result['morpho_featureID'] = [cellid2loc[x] for x in data.cells.cell_name]
        if 0 not in feature.index:
            feature.loc[0,:] = [0]*feature.shape[1]
        tmp = feature.loc[list(data.tl.result['morpho_featureID']), :]
        tmp.index = data.cells.cell_name
        data.tl.result['cell_seg_feature'] = tmp
        return cellid2loc

    def test_image_feature(self, gem_path, image_path):
        from PIL import Image
        from PIL import ImageFile
        #plt.rc('font',family='Arial')
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        Image.MAX_IMAGE_PIXELS = None
        
        im=Image.open(image_path)
        im = np.array(im)
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(im, connectivity=8)
        print('start to extract feature')
        feature = self.main(im)
        feature.to_csv('./im_feature.csv')
        data = st.io.read_gem(gem_path, bin_type = 'cell_bins')
        cellID2loc = self.mapping_labels(data, labels, feature)
        return data, feature

if __name__ == '__main__':
    image_path = '/jdfssz2/ST_BIOINTEL/P20Z10200N0039/06.groups/04.Algorithm_tools/caolei2/Image_feature_extract/data/cellbin_heart/L20220122023.tif'
    gem_path = '/jdfssz2/ST_BIOINTEL/P20Z10200N0039/06.groups/04.Algorithm_tools/caolei2/Image_feature_extract/data/cellbin_heart/L2_raw.adjusted.gem'
    test = ImageFeature()
    adata, feature = test.test_image_feature(gem_path, image_path)