"""
scanpy.pl.paga画的是connectivities_trees，这里可以就两个结果进行展示
"""

from traj import Traj
import matplotlib.pyplot as plt
import os
import time


class PlotClusterTraj():
    def main(self,
             con, x_raw, y_raw, ty,
             save_dir,
             save_na,
             count_thresh=0,
             eps_co=3,
             check_surr_co=0.75,  # 0.75
             choose_ty=None,
             type_traj='curve',
             lower_thresh_not_equal=0.5,
             show_scatter=True,
             seed_val=0,
             num_legend_per_col=12,
             tick_step=2500,
             spot_alpha=0.7,
             spot_size=3,
             line_alpha=1,
             line_width_co=1,
             line_color='#7570b3',
             uni_lwidth=False,
             text_size=5,
             n_per_inter=100,
             dpi_save=500):

        # TODO: 描述
        # TODO: 对输入进行断言

        # generating data for plotting
        traj = Traj(con, x_raw, y_raw, ty)
        traj.assign_ty_rep()

        mask_keep, keep_ty = traj.filter_minority(count_thresh)
        traj.revise_con_based_on_selection(keep_ty)

        if not choose_ty is None:
            traj.revise_con_based_on_selection(choose_ty)

        traj.estimate_avg_dis()

        traj.cal_repre_x_y_by_ty(eps_co, check_surr_co)

        traj.get_con_pairs(lower_thresh_not_equal)

        # plotting
        plt.figure()

        if show_scatter:
            traj.show_scatter(spot_size, spot_alpha, seed_val, num_legend_per_col, tick_step, mask_keep)

        traj.show_ty_label(text_size, choose_ty, keep_ty)

        if type_traj == 'curve':
            traj.compute_com_traj_li()
            print([[traj.ty_rep[i] for i in li] for li in traj.com_tra_li])

            x_unknown_li_all_tra, y_unknown_li_all_tra = traj.cal_position_param_curve(n_per_inter)

            com_tra_wei_li = traj.compute_weight_on_com_tra_li()

            traj.show_curve(x_unknown_li_all_tra, y_unknown_li_all_tra, com_tra_wei_li,
                            line_alpha, line_width_co, line_color, uni_lwidth)
        else:
            x_li, y_li = traj.cal_position_param_straight()
            wei_li = traj.compute_weight_on_pairs()

            traj.show_straight(x_li, y_li, wei_li,
                               line_alpha, line_width_co, line_color, uni_lwidth)

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.savefig(os.path.join(save_dir, save_na), dpi=dpi_save, bbox_inches='tight')
        plt.close()

    def test(self):
        def acquire_data(path, use_rep, group):
            """获取用于画图的数据"""
            adata = anndata.read_h5ad(path)

            # 选用邻居点描述每个点，来计算点和点之间的距离和权重（'PAGA Connectivities'?）, 用于paga的计算，保存于.obsp中
            scanpy.pp.neighbors(adata, use_rep=use_rep)

            # 计算每群之间的PAGA连接性，轨迹结构，保存于.uns['paga']中,其中connectivities表示paga的结果，connectivities_tree表示PAGA_TREE的结果
            # connectivities是无向的对角矩阵，connectivities_tree是无向的非对角矩阵

            # print('current ele', adata.obs['pred'].tolist())
            adata.obs[group] = adata.obs[group].astype('category')  # 以标号形式存储：从0开始，逐一递增  # categorical datatype在dataframe中才有用，加入numpy中还是原来类型

            # print('transfered to categorical', adata.obs['pred'].tolist())

            # adata.obs['pred'].cat.codes.values

            scanpy.tl.paga(adata, groups=group)  # 转移矩阵的索引对应categories标号，即adata.obs['pred']目前的存储值
            return adata

        import anndata
        import scanpy

        # path = 'E:/data/stereopy/DCIS.h5ad'
        # fig_dir = 'E:/ANALYSIS_ALGORITHM/cell_trajectory_analysis/plot_cluster_traj_result/DCIS'
        # use_rep = 'emb'
        # group = 'pred'

        # 肺癌数据
        path = 'E:/data/stereopy/ToSummer/lung_cancer/lung_cancer_pca_leiden.h5ad'
        fig_dir = 'E:/ANALYSIS_ALGORITHM/cell_trajectory_analysis/plot_cluster_traj_result/lung_cancer'
        use_rep = 'X_pca'
        group = 'leiden'

        # # 成熟鼠脑
        # path = 'E:/data/stereopy/ToSummer/mouse_brain/mouse_brain_pca_leiden.h5ad'
        # fig_dir = 'E:/ANALYSIS_ALGORITHM/cell_trajectory_analysis/plot_cluster_traj_result/mouse_brain'
        # use_rep = 'X_pca'
        # group = 'celltype_pred'

        # # E16.5鼠胚胎 整张
        # path = 'E:/data/stereopy/PRO000000271_EPT000000272_SPT000000097.h5ad'
        # use_rep = 'X_pca'
        # group = 'celltype'
        # fig_dir = 'E:/ANALYSIS_ALGORITHM/cell_trajectory_analysis/plot_cluster_traj_result/mouse_embryo_whole_cellbin/'

        # E16.5鼠胚胎 脑
        # path = 'E:/data/stereopy/PRO000000271_EPT000000272_SPT000000098.h5ad'
        # use_rep = 'X_pca'
        # group = 'celltype'
        # fig_dir = 'E:/ANALYSIS_ALGORITHM/cell_trajectory_analysis/plot_cluster_traj_result/mouse_embryo_brain_cellbin/'

        # 生成需要的数据，先用adata的形式
        adata = acquire_data(path, use_rep, group)
        print('data size', adata.X.shape)

        # 用scanpy测试: 和目前画图结果不同
        # fig = plt.figure(figsize=(4, 4))
        # ax = fig.subplots(1, 1)
        # scanpy.pl.paga(adata, solid_edges='connectivities_tree', threshold=0.5, ax=ax)
        # fig.savefig(os.path.join(fig_dir, 'paga_plt.png'), dpi=300, bbox_inches='tight')
        # plt.close()

        # x_raw = adata.obsm['spatial'][:, 0]  # arr: (n_cell,)
        # y_raw = adata.obsm['spatial'][:, 1]
        x_raw = adata.obsm['spatial_stereoseq']['X'].to_numpy()
        y_raw = adata.obsm['spatial_stereoseq']['Y'].to_numpy()

        ty = adata.obs[group].to_numpy()  # arr: (n_cell,) categorical
        con = adata.uns['paga']['connectivities'].todense()  # arr (n_clus, n_clus)
        con_tree = adata.uns['paga']['connectivities_tree'].todense()

        # plt.figure()
        # plt.imshow(con_tree)
        # plt.colorbar()
        # plt.savefig(os.path.join(fig_dir, 'con_tree_result.tif'))
        # plt.close()

        self.main(con_tree, x_raw, y_raw, ty, fig_dir, 'test.tif',
                  lower_thresh_not_equal=0.95,
                  num_legend_per_col=20,
                  choose_ty=['Mid-/hindbrain and spinal cord neuron', 'Endothelial cell', 'Cardiomyocyte'],
                  count_thresh=100,
                  seed_val=1,  # 2
                  eps_co=30, check_surr_co=20,
                  spot_size=0.1, type_traj='curve')  # choose_ty=[1, 3, 9, 0, 13, 19]


if __name__ == '__main__':
    t0 = time.time()

    pcj = PlotClusterTraj()
    pcj.test()

    t1 = time.time()

    print('t', t1-t0)




