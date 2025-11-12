import numpy as np
import gc
import os

import matplotlib.pyplot as plt

from .helper import filter_rows_cols, to_dense_array


class RegisPlotter:
    def __init__(self, num_cols=4, dpi_val=1000):
        self._num_cols = num_cols
        self._dpi_val = dpi_val

    @property
    def num_cols(self):
        return self._num_cols
    
    @num_cols.setter
    def num_cols(self, num_cols):
        self._num_cols = num_cols
    
    @property
    def dpi_val(self):
        return self._dpi_val
    
    @dpi_val.setter
    def dpi_val(self, dpi_val):
        self._dpi_val = dpi_val

    def _define_cells(self, length):
        if length % self.num_cols == 0:
            num_rows = length // self.num_cols
        else:
            num_rows = length // self.num_cols + 1
        return num_rows, self.num_cols

    def _title_pre(self, i, alphali, tyscoreli):
        # 子图表标题
        if alphali is None or tyscoreli is None:
            pass
        else:
            if i >= 1:
                plt.title('alpha: {}, score: {}'.format(alphali[i - 1], round(float(tyscoreli[i - 1]), 3)))
            else:
                plt.title('')

    def _title_next(self, slicesl, i, alphali, tyscoreli):
        if alphali is None or tyscoreli is None:
            pass
        else:
            if not i == len(slicesl) - 1:
                plt.title('alpha: {}, score: {}'.format(alphali[i - 1], round(float(tyscoreli[i - 1]), 3)))
            else:
                plt.title('')

    def _plot_clabel(self, slicesl, lay_type, i, filter_by_label, anncell_cid, label_col, spatype, alphali, tyscoreli, size):
        if lay_type == 'to_pre':
            if i == 0:
                slice, _ = filter_rows_cols(slicesl[i], slicesl[i + 1], filter_by_label=filter_by_label, label_col=label_col)
            else:
                _, slice = filter_rows_cols(slicesl[i - 1], slicesl[i], filter_by_label=filter_by_label, label_col=label_col)

            clist = [anncell_cid[anno_cell] for anno_cell in slice.obs[label_col].tolist()]
            plt.scatter(slice.obsm[spatype][:, 0], slice.obsm[spatype][:, 1], linewidths=0, c=clist, cmap='rainbow', s=size)
            plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='box')

            self._title_pre(i, alphali, tyscoreli)
        else:
            if i == len(slicesl) - 1:
                _, slice = filter_rows_cols(slicesl[i - 1], slicesl[i], filter_by_label=filter_by_label, label_col=label_col)
            else:
                slice, _ = filter_rows_cols(slicesl[i], slicesl[i + 1], filter_by_label=filter_by_label, label_col=label_col)

            clist = [anncell_cid[anno_cell] for anno_cell in slice.obs[label_col].tolist()]
            plt.scatter(slice.obsm[spatype][:, 0], slice.obsm[spatype][:, 1], linewidths=0, c=clist, cmap='rainbow', s=size)
            plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='box')

            self._title_next(slicesl, i, alphali, tyscoreli)

    def _plot_sum_gene(self, slicesl, lay_type, i, filter_by_label, label_col, spatype, alphali, tyscoreli, size):
        if lay_type == 'to_pre':
            if i == 0:
                slice, _ = filter_rows_cols(slicesl[i], slicesl[i + 1], filter_by_label=filter_by_label, label_col=label_col)
            else:
                _, slice = filter_rows_cols(slicesl[i - 1], slicesl[i], filter_by_label=filter_by_label, label_col=label_col)

            clist = np.array(slice.X.todense().sum(axis=1)).squeeze()
            plt.scatter(slice.obsm[spatype][:, 0], slice.obsm[spatype][:, 1], linewidths=0, c=clist, cmap='rainbow', s=size)
            plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='box')

            self._title_pre(i, alphali, tyscoreli)
        else:
            if i == len(slicesl) - 1:
                _, slice = filter_rows_cols(slicesl[i - 1], slicesl[i], filter_by_label=filter_by_label, label_col=label_col)
            else:
                slice, _ = filter_rows_cols(slicesl[i], slicesl[i + 1], filter_by_label=filter_by_label, label_col=label_col)

            clist = np.array(slice.X.todense().sum(axis=1)).squeeze()
            plt.scatter(slice.obsm[spatype][:, 0], slice.obsm[spatype][:, 1], linewidths=0, c=clist, cmap='rainbow', s=size)
            plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='box')

            self._title_next(slicesl, i, alphali, tyscoreli)

    def _plot_pi_max(self, slicesl, lay_type, i, filter_by_label, label_col, pili, spatype,  alphali, tyscoreli, size):
        if lay_type == 'to_pre':
            if i == 0:
                pass
            else:
                _, slice = filter_rows_cols(slicesl[i - 1], slicesl[i], filter_by_label=filter_by_label, label_col=label_col)

                pi_pre = to_dense_array(pili[i - 1])

                pi_max = np.max(pi_pre, axis=0)

                del pi_pre
                gc.collect()

                plt.scatter(slice.obsm[spatype][:, 0], slice.obsm[spatype][:, 1], linewidths=0, c=pi_max,
                            cmap='rainbow', s=size)
                plt.colorbar()
                plt.gca().set_aspect('equal', adjustable='box')
            self._title_pre(i, alphali, tyscoreli)
        else:
            if i == len(slicesl) - 1:
                pass
            else:
                slice, _ = filter_rows_cols(slicesl[i], slicesl[i + 1], filter_by_label=filter_by_label, label_col=label_col)

                pi = to_dense_array(pili[i])

                pi_max = np.max(pi, axis=1)

                del pi
                gc.collect()

                plt.scatter(slice.obsm[spatype][:, 0], slice.obsm[spatype][:, 1], linewidths=0, c=pi_max,
                            cmap='rainbow',
                            s=size)
                plt.colorbar()
                plt.gca().set_aspect('equal', adjustable='box')

            self._title_next(slicesl, i, alphali, tyscoreli)

    def _plot_num_anch(self, slicesl, lay_type, i, filter_by_label, label_col, pili, spatype, alphali, tyscoreli, size):
        if lay_type == 'to_pre':
            if i == 0:
                pass
            else:
                _, slice = filter_rows_cols(slicesl[i - 1], slicesl[i], filter_by_label=filter_by_label, label_col=label_col)

                pi_pre = to_dense_array(pili[i - 1])
                num_anch = np.sum((pi_pre > 0), axis=0)

                del pi_pre
                gc.collect()

                plt.scatter(slice.obsm[spatype][:, 0], slice.obsm[spatype][:, 1], linewidths=0, c=num_anch, cmap='rainbow',
                            s=size)
                plt.colorbar()
                plt.gca().set_aspect('equal', adjustable='box')

            self._title_pre(i, alphali, tyscoreli)
        else:
            if i == len(slicesl) - 1:
                pass
            else:
                slice, _ = filter_rows_cols(slicesl[i], slicesl[i + 1], filter_by_label=filter_by_label, label_col=label_col)

                pi = to_dense_array(pili[i])

                num_anch = np.sum((pi > 0), axis=1)

                del pi
                gc.collect()

                plt.scatter(slice.obsm[spatype][:, 0], slice.obsm[spatype][:, 1], linewidths=0, c=num_anch,
                            cmap='rainbow', s=size)
                plt.colorbar()
                plt.gca().set_aspect('equal', adjustable='box')

            self._title_next(slicesl, i, alphali, tyscoreli)

    def _plot_str_anch(self, slicesl, lay_type, i, filter_by_label, spatype, pili, label_col, anncell_cid, alphali, tyscoreli, size):
        if lay_type == 'to_pre':
            if i == 0:
                pass
            else:
                sliceA, sliceB = filter_rows_cols(slicesl[i - 1], slicesl[i], filter_by_label=filter_by_label, label_col=label_col)

                pi_pre = to_dense_array(pili[i - 1])

                # find out the spots on sliceB, pointed from anchors from sliceA
                plt_spa_arr = sliceB.obsm[spatype][np.argmax(pi_pre, axis=1), :]
                plt_anno_arr = np.array(sliceA.obs[label_col])

                # remove spots without any actual anchors
                mask_1d = (np.max(pi_pre, axis=1) > 0)

                del pi_pre
                gc.collect()

                plt_spa_arr = plt_spa_arr[mask_1d, :]
                plt_anno_arr = plt_anno_arr[mask_1d]

                # plot
                clist = [anncell_cid[anno_cell] for anno_cell in plt_anno_arr]
                plt.scatter(plt_spa_arr[:, 0], plt_spa_arr[:, 1], linewidths=0, c=clist, cmap='rainbow', s=size)
                plt.colorbar()
                plt.gca().set_aspect('equal', adjustable='box')

            self._title_pre(i, alphali, tyscoreli)

        else:
            if i == len(slicesl) - 1:
                pass
            else:
                sliceA, sliceB = filter_rows_cols(slicesl[i], slicesl[i + 1], filter_by_label=filter_by_label, label_col=label_col)

                # find out the spots on sliceA, pointed from anchors from sliceB
                pi = to_dense_array(pili[i])

                plt_spa_arr = sliceA.obsm[spatype][np.argmax(pi, axis=0), :]
                plt_anno_arr = np.array(sliceB.obs[label_col])

                # remove spots without any actual anchors
                mask_1d = (np.max(pi, axis=0) > 0)

                del pi
                gc.collect()

                plt_spa_arr = plt_spa_arr[mask_1d, :]
                plt_anno_arr = plt_anno_arr[mask_1d]

                # plot
                clist = [anncell_cid[anno_cell] for anno_cell in plt_anno_arr]
                plt.scatter(plt_spa_arr[:, 0], plt_spa_arr[:, 1], c=clist, linewidths=0, cmap='rainbow', s=size)
                plt.colorbar()
                plt.gca().set_aspect('equal', adjustable='box')

            self._title_next(slicesl, i, alphali, tyscoreli)

    def _plot_weight(self, slicesl, lay_type, i, filter_by_label, label_col, spatype, ali, bli, alphali, tyscoreli, size):
        if lay_type == 'to_pre':
            if i == 0:
                pass
            else:
                sliceA, sliceB = filter_rows_cols(slicesl[i-1], slicesl[i], filter_by_label=filter_by_label, label_col=label_col)

                del sliceA
                gc.collect()

                plt.scatter(sliceB.obsm[spatype][:, 0], sliceB.obsm[spatype][:, 1], linewidths=0, c=bli[i-1],
                            cmap='rainbow', s=size)  # b_wei_abs
                plt.colorbar()
                plt.gca().set_aspect('equal', adjustable='box')
            self._title_pre(i, alphali, tyscoreli)
        elif lay_type == 'to_next':
            if i == len(slicesl) - 1:
                pass
            else:
                sliceA, sliceB = filter_rows_cols(slicesl[i], slicesl[i+1], filter_by_label=filter_by_label, label_col=label_col)

                del sliceB
                gc.collect()

                plt.scatter(sliceA.obsm[spatype][:, 0], sliceA.obsm[spatype][:, 1], linewidths=0, c=ali[i],
                            cmap='rainbow', s=size)
                plt.colorbar()
                plt.gca().set_aspect('equal', adjustable='box')
            self._title_next(slicesl, i, alphali, tyscoreli)

    def plot_scatter_by_grid(
        self, slicesl, anncell_cid, ali, bli, pili, tyscoreli, alphali, 
        figsize, lay_type, ctype, spatype, filter_by_label, label_col, sdir, size, invert_y=True
    ):
        """

        Parameters
        ----------
        slicesl: slicesl[i]: annData
        anncell_cid: dic
        pili: pili[i]: np.array
        tyscoreli: tyscoreli[i]: float
        alphali: alphali[i]: float
        figsize: tuple
        type: 'to_pre', or 'to_next'
        ctype: 'cell_label', 'num_anchors', 'pi_max', or 'strong_anch'
        spatype: 'spatial', 'spatial_rigid', 'spatial_elas', ...
        filter_by_label: True, or False
        sdir: dir

        Returns
        -------

        """

        assert type(ctype) == str, "ctype is not str"
        assert ctype in ['cell_label', 'num_anchors', 'pi_max', 'strong_anch', 'weight', 'sum_gene']

        assert lay_type is None or type(lay_type) == str, "lay_type is not a str or NoneType"
        assert lay_type in ['to_pre', 'to_next'], "lay_type not in ['to_pre', 'to_next']"

        if ctype == 'weight' and lay_type == 'to_pre':
            assert bli is not None, "ctype is weight, and lay_type is to_pre, while bli is None"

        if ctype == 'weight' and lay_type == 'to_next':
            assert ali is not None, "ctype is weight, and lay_type is to_next, while ali is None"

        assert spatype in ['spatial', 'spatial_rigid', 'spatial_elas'], "spatype not in ['spatial', 'spatial_rigid', 'spatial_elas']"
        assert list(set([spatype in slicesl[i].obsm.keys() for i in range(len(slicesl))])) == [True], "Not all slicesl element has {} column".format(spatype)
        assert type(filter_by_label) == bool, "Type of filter_by_label is not bool"
        if not sdir is None:
            assert type(sdir) == str, "Type of sdir is not str, or None"
        assert type(size) in [float, int], "Type of size not in float or string"
        assert size > 0, "size not over 0"

        nrows, ncols = self._define_cells(len(slicesl))

        # fig, _ = plt.subplots(nrows, ncols, figsize=figsize, dpi=self.dpi_val, sharex='all', sharey='all')  # (行数，列数)， （width， height）
        fig, _ = plt.subplots(nrows, ncols, figsize=figsize, dpi=self.dpi_val, sharey='all')  # (行数，列数)， （width， height）
        fig.tight_layout()

        # for i, slice in enumerate(slicesl):
        for i in range(nrows * ncols):
            ax = plt.subplot(nrows, ncols, i + 1)
            if i >= len(slicesl):
                ax.axis('off')
                continue
            if ctype == 'cell_label':
                    self._plot_clabel(slicesl, lay_type, i, filter_by_label, anncell_cid, label_col, spatype, alphali, tyscoreli, size)  # todo: 去掉filter_by_label

            elif ctype == 'sum_gene':
                    self._plot_sum_gene(slicesl, lay_type, i, filter_by_label, label_col, spatype, alphali, tyscoreli, size)  # todo: 去掉filter_by_label

            elif ctype == 'pi_max':
                try:
                    self._plot_pi_max(slicesl, lay_type, i, filter_by_label, label_col, pili, spatype,  alphali, tyscoreli, size)
                except:
                    pass
            elif ctype == 'num_anchors':
                try:
                    self._plot_num_anch(slicesl, lay_type, i, filter_by_label, label_col, pili, spatype, alphali, tyscoreli, size)
                except:
                    pass
            elif ctype == 'strong_anch':  # strongest anchor 画上一片各点最强的锚所对应的本片的位置
                try:
                    self._plot_str_anch(slicesl, lay_type, i, filter_by_label, spatype, pili, label_col, anncell_cid, alphali, tyscoreli, size)
                except:
                    pass
            elif ctype == 'weight':
                    self._plot_weight(slicesl, lay_type, i, filter_by_label, label_col, spatype, ali, bli, alphali, tyscoreli, size)
        if invert_y:
            ax.invert_yaxis()
        # if sdir is None:
        #     plt.show()
        # else:
        #     if not os.path.isdir(sdir):
        #         os.makedirs(sdir)
        #     plt.savefig(os.path.join(sdir, spatype + '_' + ctype+'.jpg'))

        #     plt.close()
        
        return fig