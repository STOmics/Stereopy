import random
import glob
# import hdf5plugin
import os
from natsort import natsorted
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import scanpy as sc
# from scanpy import logging as logg
import torch
import torch.optim as optim
from tqdm import tqdm
from anndata import AnnData

from stereo.log_manager import logger

from .spaseg_model import SegNet
from ._utils import get_3d_expMatrix, get_max_H_W, seed_torch, merge_outlier, add_embedding, cal_metric


# sc.logging.print_header()
# sc.settings.verbosity = 3
# sc.set_figure_params(facecolor="white", figsize=(6, 6), dpi_save=300, fontsize=10, vector_friendly=True)
# sc._settings.ScanpyConfig.figdir = opt.out_file_path

class SpaSEG():
    def __init__(
        self,
        adata: Sequence[AnnData],
        use_gpu: bool = False,
        device: Optional[str] = "cuda",
        seed: int = 1029,
        input_dim: int = 15,
        nChannel: int = 15,
        output_dim: int = 15,
        nConv: int = 2,
        lr: float = 0.002,
        weight_decay: float = 1e-5,
        pretrain_epochs: int = 400,
        iterations: int = 2100,
        sim_weight: float = 0.4,
        con_weight: float = 0.7,
        min_label: int = 7,
        spot_size: int = None,
        result_prefix: str = "SpaSEG",
    ):

        self.adata = adata
        self.use_gpu = use_gpu
        self.device = device
        self.seed = seed
        self.input_dim = input_dim
        self.nChannel = nChannel
        self.output_dim = output_dim
        self.nConv = nConv
        self.lr = lr
        self.weight_decay = weight_decay
        self.pretrain_epochs = pretrain_epochs
        self.iterations = iterations
        self.sim_weight = sim_weight
        self.con_weight = con_weight
        self.min_label = min_label
        self.spot_size = spot_size
        self.result_prefix = result_prefix

    def _prepare_data(self):
        #if isinstance(self.adata, Sequence):
        # initialize input matrix list and unified H and W for input image-like 3-d matrix
        input_st_mxt_list = []
        H, W = get_max_H_W(self.adata)
        for adata in self.adata:
            input_st_mxt, col, row = get_3d_expMatrix(adata, self.input_dim, H, W)
            input_st_mxt_list.append(input_st_mxt)

        input_st_mxt = np.stack(input_st_mxt_list, axis=0)

        return input_st_mxt, H, W


    def _train(self,
               st_mxt: np.ndarray):
            # fix seed
            seed_torch(self.seed)

            use_cuda = torch.cuda.is_available()
            if use_cuda and self.use_gpu:
                device = self.device
                logger.info('\nPut the data and model into GPU')
            else:
                device = "cpu"
                logger.info('\nPut the data and model into CPU')

            # feed the input matrix to GPU
            logger.info(f'input matrix shape: {st_mxt.shape}')
            logger.info(f'feed the input matrix to {device}')
            data = torch.from_numpy(st_mxt.astype('float32'))
            data = data.to(device)

            im = np.transpose(st_mxt, [0, 2, 3, 1])
            np.set_printoptions(threshold=np.inf)

            logger.info(f'create the model and put it into {device}')
            model = SegNet(input_dim=self.input_dim, nChannel=self.nChannel, output_dim=self.output_dim, nConv=self.nConv)
            model = model.to(device)

            maxIter = self.iterations
            logger.info(f'Start training with {maxIter} iterations')
            model.train()

            # similarity loss definition
            loss_CE = torch.nn.CrossEntropyLoss()

            loss_MSE = torch.nn.MSELoss()

            # edge loss definition
            loss_edge_vertical = torch.nn.L1Loss(reduction='mean')
            loss_edge_horizontal = torch.nn.L1Loss(reduction='mean')

            vertical_target = torch.zeros(data.shape[0], im.shape[1] - 1, im.shape[2], self.output_dim)
            horizontal_target = torch.zeros(data.shape[0], im.shape[1], im.shape[2] - 1, self.output_dim)
            if use_cuda:
                vertical_target = vertical_target.to(device)
                horizontal_target = horizontal_target.to(device)

            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            final_target = None
            iterator = tqdm(range(maxIter))
            for batch_idx in iterator:
                optimizer.zero_grad()
                output = model(data)
                # print("output shape is: {}".format(output.shape), end = "\r")

                origin_data = data.permute(0, 2, 3, 1).contiguous().view(data.shape[0], -1, self.output_dim)
                output1 = output.permute(0, 2, 3, 1).contiguous().view(data.shape[0], -1, self.output_dim)
                output = output1.reshape(output1.shape[0] * output1.shape[1], -1)
                np.set_printoptions(threshold=np.inf)
                ignore, targetxx = torch.max(output, 1)
                im_target = targetxx.data.cpu().numpy()

                if batch_idx < 2000:
                    ignore, target = torch.max(output, 1)
                    im_target = target.data.cpu().numpy()
                    final_target = im_target
                    n_labels = len(np.unique(im_target))
                elif batch_idx <= 2100:
                    ignore, targetxxx = torch.max(output, 1)
                    target = merge_outlier(targetxxx, data, im, device)
                    im_target = target.data.cpu().numpy()
                    final_target = im_target
                    n_labels = len(np.unique(im_target))

                outputEG = output.reshape((data.shape[0], im.shape[1], im.shape[2], self.output_dim))
                EG_ver = outputEG[:, 1:, :, :] - outputEG[:, 0:-1, :, :]
                EG_hor = outputEG[:, :, 1:, :] - outputEG[:, :, 0:-1, :]
                leg_ver = loss_edge_vertical(EG_ver, vertical_target)
                leg_hor = loss_edge_horizontal(EG_hor, horizontal_target)
                if batch_idx < self.pretrain_epochs:
                    loss = loss_MSE(output.contiguous().view(-1), origin_data.contiguous().view(-1))
                    # print("pretrain loss for each epoch:{:.3}".format(loss), end = "\r")
                else:
                    sim_loss = loss_CE(output, target)
                    con_loss = leg_ver + leg_hor
                    loss = self.sim_weight * sim_loss + self.con_weight * con_loss
                    # print("similarity loss:{:.3}, continuity loss:{:.3}".format(sim_loss, con_loss), end = "\r")

                loss.backward()
                optimizer.step()
                # print(batch_idx, '/', maxIter, ':', n_labels, loss.item(), end = "\r")

                if n_labels <= self.min_label:
                    logger.info("nLabels {} reached minLabels {}.".format(n_labels,self.min_label))
                    iterator.close() 
                    break

            SpaSEG_embedding = output1.cpu().detach().numpy()

            return final_target, SpaSEG_embedding

    def _add_embedding(self, H, W, embedding, n_batch):
        self.adata = [self._map_embedding(adata, H, W, embedding[i], self.nChannel) for adata, i in
         zip(self.adata, range(0, n_batch))]

    # def _add_seg_label(self, label, n_batch, H, W, barcode_index):
    #     lab_im = label.reshape((n_batch, H, W))
    #     unique_lab = np.unique(lab_im)
    #     colormap = self._color_map(unique_lab)
    #     index_map = {dis_ind:con_ind for dis_ind, con_ind in zip(unique_lab, range(0, len(unique_lab)))}
    #     self.adata = [self._spot_mapping(adata, lab_im[i], barcode_index, colormap, index_map) for adata, i in zip(self.adata, range(0, n_batch))]

    def _add_seg_label(self, label, H, W):
        lab_im = label.reshape((len(self.adata), H, W))
        unique_lab = np.unique(lab_im)
        index_map = {dis_ind: con_ind for con_ind, dis_ind in enumerate(unique_lab)}
        
        def __apply_func(row, adata_idx):
            lab_discrete = lab_im[adata_idx][int(row['array_row']), int(row['array_col'])]
            lab_continuous = index_map[lab_discrete]
            lab_discrete += 1
            lab_continuous += 1
            return str(lab_discrete), str(lab_continuous)
        
        col1 = f'{self.result_prefix}_discrete_clusters'
        col2 = f'{self.result_prefix}_clusters'
        for i, adata in enumerate(self.adata):
            adata.obs[[col1, col2]] = adata.obs.apply(__apply_func, axis=1, result_type='expand', args=(i,))
            col1_unique = natsorted(adata.obs[col1].unique())
            adata.obs[col1] = pd.Categorical(values=adata.obs[col1], categories=col1_unique)
            col2_unique = natsorted(adata.obs[col2].unique())
            adata.obs[col2] = pd.Categorical(values=adata.obs[col2], categories=col2_unique)

    def _cal_metrics(self, true_labels_column=None):
        pred_labels_column = f'{self.result_prefix}_clusters'
        self.adata = [cal_metric(adata, pred_labels_column, true_labels_column, self.result_prefix) for adata in self.adata]

    # def _save_result(self, result_dir, sample_id_list, save_umap):
    #     # check whether the metrics result.csv file exits
    #     result_file = "{}/result.csv".format(result_dir)
    #     if os.path.exists(result_file):
    #         result_df = pd.read_csv(result_file, index_col=0)
    #     else:
    #         metrics = self.adata[0].uns['metrics']
    #         result_df = pd.DataFrame(index=sample_id_list, columns=metrics.keys())
    #     for adata, sample_id in zip(self.adata, sample_id_list):
    #         metrics = adata.uns['metrics']
    #         for key, val in metrics.items():
    #             result_df.loc[sample_id, key] = val
    #         if self.spot_size:
    #             spot_size = self.spot_size
    #         else:
    #             try:
    #                 spot_size = adata.uns["spot_size"]
    #             except Exception as e:
    #                 spot_size = 100 ### using the default spot size for 10X Visium data
    #                 print("Either spot_size should be initialize in SpaSEG object or exit in adata.uns['spot_size'],\nusing the default spot size 100!")
    #         sc.pl.spatial(adata, color="SpaSEG_clusters", spot_size=spot_size, title="SpaSEG",
    #                       save='SpaSEG_{}.pdf'.format(sample_id), show=False, frameon=False)
    #         # whether save the h5ad file or not
    #         adata.write_h5ad("{}/SpaSEG_{}.h5ad".format(result_dir, sample_id))
    #     result_df.to_csv("{}/result.csv".format(result_dir))

    #     if save_umap:
    #         batch_umap_plot(self.adata, sample_id_list)


    def _map_embedding(self, adata, H, W, embedding, nChannel):
        SpaSEG_embedding = embedding.reshape((H, W, nChannel))
        col = adata.obs['array_col'].values.astype(int)
        row = adata.obs['array_row'].values.astype(int)
        shape = adata.obsm['X_pca'].shape

        poss = zip(col, row)
        SpaSEG_pca = np.zeros(shape)

        for i, idx in enumerate(poss):
            SpaSEG_pca[i, :] = SpaSEG_embedding[idx[1], idx[0], :]

        adata.obsm["SpaSEG_embedding"] = SpaSEG_pca

        return adata

    # def _color_map(self, cluster_label):
    #     vega_10 = sc.pl.palettes.vega_10_scanpy
    #     vega_20 = sc.pl.palettes.vega_20_scanpy
    #     zeileis_28 = sc.pl.palettes.zeileis_28
    #     godsnot_102 = sc.pl.palettes.godsnot_102

    #     length = len(cluster_label)
    #     if length <= 10:
    #         palette = vega_10
    #     elif length <= 20:
    #         palette = vega_20
    #     elif length <= 28:
    #         palette = zeileis_28
    #     elif length <= len(godsnot_102):  # 103 colors
    #         palette = godsnot_102
    #     else:
    #         palette = ['grey' for _ in range(length)]
    #         logger.info(
    #             f'the clusters has more than 103 categories. Uniform '
    #             "'grey' color will be used for all categories."
    #         )
    #     colormap = {label: color for label, color in zip(cluster_label, palette[:length])}

    #     return colormap


    # def _spot_mapping(self, adata, lab_im, barcode_index, colormap, index_map):
    #     cols_bc = adata.obs['array_col'].copy()
    #     rows_bc = adata.obs['array_row'].copy()
    #     x = cols_bc.to_frame().reset_index()
    #     y = rows_bc.to_frame().reset_index()
    #     row_col = y.merge(x, on=barcode_index)
    #     row_col['array_row'] = row_col['array_row'].astype(int)
    #     row_col['array_col'] = row_col['array_col'].astype(int)

    #     pt_label = []
    #     for i, i_row in row_col.iterrows():
    #         pt_label.append(lab_im[i_row['array_row'], i_row['array_col']])

    #     row_col['SpaSEG_discrete_clusters'] = pd.Categorical(pt_label)
    #     xx = row_col[[barcode_index, 'SpaSEG_discrete_clusters']].set_index(barcode_index)

    #     xx.index.name = None
    #     # print (xx)
    #     adata.obs['SpaSEG_discrete_clusters'] = xx.squeeze()

    #     adata.obs['SpaSEG_clusters'] = adata.obs['SpaSEG_discrete_clusters'].apply(lambda x: index_map[x])
    #     discrete_lab = np.unique(adata.obs['SpaSEG_discrete_clusters'])

    #     adata.uns["SpaSEG_clusters_colors"] = [colormap[i] for i in discrete_lab]
    #     return adata