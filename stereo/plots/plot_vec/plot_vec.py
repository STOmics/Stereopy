import os
import time

import numpy as np
import matplotlib.pyplot as plt

from stereo.plots.plot_vec.vec import Vec
from stereo.plots.plot_base import PlotBase


class PlotVec(PlotBase):

    # TODO：加入输入的边界条件
    # TODO: 把画图的代码单独取出, 其余的结构性代码作为main，放在plot_vec中

    def plot_vec(
            self,
            x_raw,
            y_raw,
            ty_raw,
            ptime,
            fig_dir,
            type='vec',
            fig_name='fig.tif',
            count_thresh=0,
            tick_step=4000,
            line_len_co=1,
            vec_alpha=1,
            line_width=0.0025,
            density=2,
            background=None,
            background_alpha=0.5,
            num_pix=50,
            filter_type='gauss',
            sigma_val=0.4,
            radius_val=1,
            scatter_s=1,
            seed_val=1,
            num_legend_per_col=12,
            dpi_val=1000
    ):
        assert len(x_raw.shape) == 1 and x_raw.shape[0] == y_raw.shape[0] == ptime.shape[0], \
            "input has wrong array shape"

        # 预处理，获得矩阵数据需要的输入
        vec = Vec()
        x_raw, y_raw, ty_raw, ptime = vec.filter_minority(ty_raw, count_thresh, x_raw, y_raw, ty_raw, ptime)

        vec.preprocess(x_raw, y_raw, num_pix)
        print('preprocessed')

        # 生成画图用的矩阵数据
        plt_avg_ptime = vec.gen_arr_for_mean(ptime)
        plt_avg_ptime_fil = vec.filter(plt_avg_ptime, filter_type, sigma_val, radius_val)

        plt_common_ty = vec.gen_arr_for_common(ty_raw)
        print('most common type in each pixel calculated.')

        u, v = vec.cal_param(plt_avg_ptime_fil)
        print('u, v calculated.')

        mask_nan = np.isnan(u) | np.isnan(v) | (u == 0) | (v == 0)
        u[mask_nan] = np.nan
        v[mask_nan] = np.nan

        vec.plot_line(x_raw, y_raw, ty_raw, plt_common_ty, u, v,
                      type, background, background_alpha, scatter_s,
                      seed_val, num_legend_per_col,
                      line_len_co, vec_alpha, line_width, density,
                      tick_step, dpi_val, fig_dir, fig_name)

    def plot_time_scatter(self, group='leiden', fig_dir="./", fig_name="test_plot_time_scatter.tif"):
        data = self.stereo_exp_data
        x_raw = data.position[:, 0]
        y_raw = data.position[:, 1]
        # x_raw = adata.obsm['spatial_stereoseq']['X'].to_numpy()
        # y_raw = adata.obsm['spatial_stereoseq']['Y'].to_numpy()

        data.cells[group] = data.cells[group].astype('category')
        ptime = data.tl.result['dpt_pseudotime']

        plt.figure()
        plt.scatter(x_raw, y_raw, s=1, c=ptime, linewidths=0, cmap='rainbow')
        plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(os.path.join(fig_dir, fig_name), dpi=2000)
        plt.close()

    def test(self):
        def preprocess_data(adata, use_rep):
            """获取用于画图的数据"""

            # 选用邻居点描述每个点，来计算点和点之间的距离和权重（'PAGA Connectivities'?）, 用于paga的计算，保存于.obsp中
            scanpy.pp.neighbors(adata, use_rep=use_rep)

            return adata

        import anndata
        import scanpy
        import numpy as np

        # E16.5鼠胚胎 整张
        # path = 'D:\projects\stereopy_dev\demo_data\PRO000000271_EPT000000272_SPT000000097\PRO000000271_EPT000000272_SPT000000097.h5ad'
        # use_rep = 'X_pca'
        # group = 'celltype'
        # fig_dir = 'D:\projects\stereopy_dev\demo_data\PRO000000271_EPT000000272_SPT000000097'

        # path = 'D:\projects\stereopy_dev\demo_data\SS200000135TL_D1\SS200000135TL_D1.tissue.h5ad'
        # data = AnnBasedStereoExpData(path)
        # use_rep = 'X_pca'
        # group = 'leiden'
        # fig_dir = 'D:\projects\stereopy_dev\demo_data\SS200000135TL_D1'

        # path = "/mnt/d/projects/stereopy_dev/demo_data/SS200000135TL_D1/SS200000135TL_D1.tissue.h5ad"
        # data = AnnBasedStereoExpData(path)

        path = "/mnt/d/projects/stereopy_dev/demo_data/SS200000135TL_D1/SS200000135TL_D1.tissue.gef"
        import stereo as st
        data = st.io.read_gef(path)
        data.tl.normalize_total(target_sum=1e4)
        data.tl.log1p()
        data.tl.highly_variable_genes(
            min_mean=0.0125, max_mean=3, min_disp=0.5, res_key='highly_variable_genes', n_top_genes=None
        )
        data.tl.scale(zero_center=False)
        data.tl.pca(use_highly_genes=True, hvg_res_key='highly_variable_genes', n_pcs=20, res_key='pca',
                    svd_solver='arpack')
        data.tl.neighbors(pca_res_key='pca', n_pcs=30, res_key='neighbors', n_jobs=2)
        data.tl.leiden(neighbors_res_key='neighbors', res_key='leiden')

        use_rep = 'X_pca'
        group = 'leiden'
        fig_dir = '/mnt/d/projects/stereopy_dev/demo_data/SS200000135TL_D1/'

        # 生成需要的数据，先用adata的形式
        # adata = anndata.read_h5ad(path)
        print('read in')

        # adata = preprocess_data(adata, use_rep)
        print(data.exp_matrix.shape)

        # 计算用于画图的基本数据类型：坐标,类型和伪时序
        x_raw = data.position[:, 0]
        y_raw = data.position[:, 1]
        # x_raw = adata.obsm['spatial_stereoseq']['X'].to_numpy()
        # y_raw = adata.obsm['spatial_stereoseq']['Y'].to_numpy()

        data.cells[group] = data.cells[group].astype('category')
        ty_raw = data.cells[group].to_numpy()
        # 肺癌：'0' # 鼠胚胎：'Spinal cord neuron',100  # 发育鼠脑：'Forebrain radial glia',100 # 鼠脑：'DGGRC2', 100
        # 需要尝试不同的细胞索引
        data.tl.result['iroot'] = np.flatnonzero(data.cells[group] == '1')[100]
        from stereo.algorithm.dpt import dpt

        dpt(data, n_branchings=0)
        ptime = data.tl.result['dpt_pseudotime']

        import os

        # # 画伪时序图
        # plt.figure()
        # plt.scatter(x_raw, y_raw, s=1, c=ptime, linewidths=0, cmap='rainbow')
        # plt.colorbar()
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.savefig(os.path.join(fig_dir, 'test_2.tif'), dpi=2000)
        # plt.close()

        self.main(x_raw, y_raw, ty_raw, ptime,
                  fig_dir,
                  type='stream',
                  line_width=0.5,
                  fig_name='test.tif',
                  background='field',
                  num_pix=50,
                  filter_type='gauss',
                  sigma_val=1,
                  radius_val=3,
                  scatter_s=0.1,
                  density=2,
                  seed_val=0,
                  num_legend_per_col=20,
                  dpi_val=2000)


if __name__ == '__main__':
    t0 = time.time()

    pv = PlotVec()
    pv.test()

    t1 = time.time()

    t = t1 - t0
    print('t', t)
