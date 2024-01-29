"""
仅包括mesh的计算，不包括画图。计算和画图放在不同模块中
"""
import math

import numpy as np

try:
    try:
        import open3d_cpu as o3d
    except ImportError:
        try:
            import open3d as o3d  # 420.5 MB
        except Exception:
            # in order to use other method, don't raise Exception
            o3d = None

    import pyvista as pv
    import pymeshfix as mf
    import pyacvd

except ImportError:
    errmsg = """
************************************************
* Some necessary modules may not be installed. *
* Please install them by:                      *
*   pip install pyvista                        *
*   pip install pymeshfix                      *
*   pip install pyacvd                         *
*   pip install open3d # or open3d-cpu         *
************************************************
    """
    raise ImportError(errmsg)

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from stereo.algorithm.algorithm_base import AlgorithmBase


class ThreeDimGroup():
    def __init__(self, xli, yli, zli, tyli, ty_name, eps_val=1.5, min_samples=8, thresh_num=10):
        """
        :param xli: List of x, with the same order of cells / bins as other iterable inputs
        :param yli: List of y, with the same order of cells / bins as other iterable inputs
        :param zli: List of z, with the same order of cells / bins as other iterable inputs
        :param tyli: List of type, with the same order of cells / bins as other iterable inputs
        :param ty_name: string of type being selected
        :param eps_val: In DBSCAN, the maximum distance between two samples for one to be considered as in the neighborhood
                        of the other. Try increase this value and min_samples if a cell type region is splitted into too
                        many subregions.
        :param min_samples: In DBSCAN, the number of samples (or total weight) in a neighborhood for a point to be
                            considered as a core point. This includes the point itself. Try increase this value and
                             eps_val if a cell type region is splitted into too many subregions. Try increase this value if
                             too many cells are excluded as outliers.
        :param thresh_num: minimum number of cells in a group to be considered to display a mesh
        """  # noqa
        self.xli = xli
        self.yli = yli
        self.zli = zli
        self.tyli = tyli
        self.ty_name = ty_name
        self.eps_val = eps_val
        self.min_samples = min_samples
        self.thresh_num = thresh_num
        if ty_name == 'all':
            # todo: 是否有必要：增加前处理：全部点云作为整体，去除异常值？郭力东老师建议：不要增加此步骤，以免掩盖tissue cut不到位出现的问题
            scatter_li = [{'x': np.array(self.xli), 'y': np.array(self.yli), 'z': np.array(self.zli), 'ty': 'all'}]
        else:
            # list, length equals to number of spatial clusters of this type
            scatter_li = self._select_split_and_remove_outlier(eps_val, min_samples,
                                                               thresh_num)

        self.scatter_li = scatter_li

    def create_mesh_of_type(self, method='march',
                            alpha=None, radii=None,
                            depth=None, width=None, scale=None, linear_fit=None, density_threshold=None,
                            mc_scale_factor=None, levelset=None,
                            tol=None):
        """
        generate mesh for each element in self.scatter_li
        :param ty_name:
        :param method:
                todo: documentation 附上示意图
                todo: 郭力东老师建议，保留march_cubes, alpha_shapes, delaunay_3d
                -------------------
               'march', 'march_cubes'

               Method parameter among 'march' and 'march_cubes' indicates using Marching Cubes Algorithm to create mesh. Marching Cubes
               voxelize a point cloud and assigns 0 / 1 to each voxel/cube, then finds the iso face of each cube that
               intersects the edges of the cube into opposite classification. A mesh is formed by the iso faces.

               It is robust to noises and lackage of points, but may add extra mesh that doesn't exist physically.

                We recommend you to browse https://www.cs.carleton.edu/cs_comps/0405/shape/marching_cubes.html for visualization
                of this algorithm. Refer to its publication for more details: https://dl.acm.org/doi/pdf/10.1145/37402.37422

                -------------------
                'alpha', 'alpha_shape'

                Method parameter among 'alpha' and 'alpha_shape' indicates using Alpha Shapes Algorithm to create mesh. Alpha
                Shapes specifies mesh of point cloud, by finding all node-node combinations that supports the sphere to
                pass them without surrounding a third point.

                Though capable of generating both concave and convex surfaces, it is not robust enough when dealing with
                non-uniformly distributed points, or relatively small hollows, since its alpha value is globally adopted,
                instead of locally optimized.

                We recommend you to read
                https://graphics.stanford.edu/courses/cs268-11-spring/handouts/AlphaShapes/as_fisher.pdf
                for a more in-depth understanding of this algorithm, or to read its publication for more details:
                https://www.cs.jhu.edu/~misha/Fall05/Papers/edelsbrunner94.pdf

                --------------------
                'ball', 'ball_pivot'

                Method parameter value among 'ball' and 'ball_pivot' indicates using Ball Pivoting Algorithm (BPA) to create mesh.
                BPA is quite closed to Alpha in terms of its process. In BPA, Three points form a triangle if a ball of
                certain radius touches them without containing any other point. Starting with a seed triangle, the ball
                pivots around an edge until it touches another point, forming another triangle. The process continues
                until all reachable edges have been tried, and then starts from another seed triangle, until all points
                are considered.

                Though capable of generating both concave and convex surfaces, it is not robust enough when dealing with
                non-uniformly distributed points, or relatively small hollows, since its alpha value is globally adopted,
                instead of locally optimized. We have also noticed some failures in mesh generation due to points adopted
                for initialization.

                Refer to its publication for more details
                https://vgc.poly.edu/~csilva/papers/tvcg99.pdf

                ---------------------
                'poisson', 'poisson_surface'

                Method parameter value among 'poisson' and 'poisson_surface' indicates using Poisson Mesh Reconstruction
                to create mesh. It divides the spatial domain by octrees, uses the sub-domains to construct the point domain
                in different ways (called indicator function), then solves the Poisson problem equation of divergence
                of gradient of indicator function equals to divergence of oriented points. This poisson equation, which is
                naturally a Partial Differential Equation, can be then transformed to a Optimization problem without restrictions
                for solution.

                Poisson Surface provides benefits of noise robustness due to its global algorithm nature, compared to other
                mesh reconstruction methods. However, cases with non-uniformly distributed points, e.g. cases with higher slice interval
                than spot intervals, poorly annotated / clustered, etc., may cast huge side effect to its performance.

                Refer to its publication for more details: https://hhoppe.com/poissonrecon.pdf

                -------------------------
                'delaunay', 'delaunay_3d'

                Method parameter value among 'delaunay' and 'delaunay_3d' indicates using Delaunay 3d Mesh Reconstruction
                to create mesh. It iteratively finds out the 3D triangulation of points with tedrahedra close
                to regular tedrahedra, using Bowyer-Watson algorithm, then extracts the surface to form a mesh. The
                output of the Delaunay triangulation is supposedly a convex hull, hence may surafce that 'overly wrap'
                points clouds, which idealy forms concave shapes.

                It is most idealized to be used in situations where the convex shape is known as prior knowledge.

                Refer to its publication for more details:
                https://www.kiv.zcu.cz/site/documents/verejne/vyzkum/publikace/technicke-zpravy/2002/tr-2002-02.pdf


        :param alpha: If method is among 'alpha' and 'alpha_shape', it is the radius of sphere that is used to construct
                      mesh. Or else if alpha is among 'delaunay' and 'delaunay_3d', alpha is the distance value to control
                      output of this filter. For a non-zero alpha value, only vertices, edges, faces, or tetrahedra
                      contained within the circumsphere (of radius alpha) will be output. Otherwise, only tetrahedra will
                      be output.
        :param radii: If method is among 'ball' and 'ball_pivot', it is the radii of the ball that are used for the
                      surface reconstruction.
        :param depth: If method is among 'poisson' and 'poisson_surface', it is the maximum depth of the tree that will
                      be used for surface reconstruction.
        :param width: If method is among 'poisson' and 'poisson_surface', it specifies the target width of the finest
                      level octree cells.
        :param scale: If method is among 'poisson' and 'poisson_surface', it specifies the ratio between the diameter of
                      the cube used for reconstruction and the diameter of the samples' bounding cube.
        :param linear_fit: If method is among 'poisson' and 'poisson_surface', if true, the reconstructor will use linear
                           interpolation to estimate the positions of iso-vertices.
        :param density_threshold: If method is among 'poisson' and 'poisson_surface', the mesh with density lower than
                                  this quantile will be filtered.
        :param mc_scale_factor: scale_factor adpoted in the Marching Cubes method, so that mc_scale_factor times of
                                maximum neighbor-wise distance equals to width of one voxel
        :param levelset: If method in 'march' and 'march_cubes', this is the iso value when generating the iso surfaces.
        :param tol: If method in 'delaunay', 'delaunay_3d', cells smaller than this will be degenerated and merged.

        :return: mesh
        """  # noqa

        mesh_li = []
        for scatter in self.scatter_li:
            if method in ['alpha', 'alpha_shape']:
                _args = {"alpha": 2.0}
                if alpha is not None:
                    _args['alpha'] = alpha
                mesh = self._create_mesh_alpha_shape(scatter, alpha=alpha)

            elif method in ['ball', 'ball_pivot']:
                _args = {"radii": [1]}
                if not radii is None:  # noqa
                    _args['radii'] = radii
                mesh = self._create_mesh_ball_pivot(scatter, radii=_args['radii'])

            elif method in ['poisson', 'poisson_surface']:
                _args = {"depth": 8, "width": 0, "scale": 1.1, "linear_fit": False, "density_threshold": None}
                if not depth is None:  # noqa
                    _args['depth'] = depth
                if not width is None:  # noqa
                    _args['width'] = width
                if not scale is None:  # noqa
                    _args['scale'] = scale
                if not linear_fit is None:  # noqa
                    _args['linear_fit'] = linear_fit
                if not density_threshold is None:  # noqa
                    _args['density_threshold'] = density_threshold

                mesh = self._create_mesh_poisson_surface(scatter,
                                                         depth=_args['depth'], width=_args['width'],
                                                         scale=_args['scale'],
                                                         linear_fit=_args['linear_fit'],
                                                         density_threshold=_args['density_threshold'])

            elif method in ['delaunay', 'delaunay_3d']:
                _args = {'alpha': 0, 'tol': 0.01}
                if not alpha is None:  # noqa
                    _args['alpha'] = alpha
                if not tol is None:  # noqa
                    _args['tol'] = tol
                mesh = self._create_mesh_delaunay(scatter, alpha=_args['alpha'], tol=_args['tol'])  # or 1.5?

            elif method in ['march', 'march_cubes']:
                _args = {"levelset": 0, "mc_scale_factor": 1}
                if not levelset is None:  # noqa
                    _args['levelset'] = 0
                if not mc_scale_factor is None:  # noqa
                    _args['mc_scale_factor'] = mc_scale_factor

                mesh = self._create_mesh_march_cubes(scatter,
                                                     mc_scale_factor=_args['mc_scale_factor'],
                                                     levelset=_args['levelset'])
            mesh_li.append(mesh)

        # mesh = self._merge_models(mesh_li)
        #
        # # post-process
        # mesh = self._remove_duplicated_mesh(mesh)
        #
        # mesh = self._fix_mesh(mesh)

        # post-process
        for i, mesh in enumerate(mesh_li):
            mesh = self._remove_duplicated_mesh(mesh)
            mesh = self._fix_mesh(mesh)
            mesh_li[i] = mesh

        mesh_all = self._merge_models(mesh_li)
        return mesh_all, mesh_li

    def uniform_re_mesh_and_smooth(self, mesh):
        """Get a smooth, uniformly meshed surface using voronoi clustering"""

        def unstruc_grid2polydata(grid):
            """
            e.g. grid = mesh.split_bodies(), polydata (mesh) = unstruc_grid2polydata(grid)
            :param grid:
            :return:
            """
            mesh = pv.wrap(grid.extract_surface())
            v = mesh.points

            # 修改face的底层数据,使之完全由三角形组成
            faces = mesh.faces
            if not mesh.is_all_triangles:
                tri_mesh = mesh.triangulate()
                faces = tri_mesh.faces
            f = np.ascontiguousarray(faces.reshape(-1, 4)[:, 1:])
            triangles = np.empty((f.shape[0], 4), dtype=pv.ID_TYPE)
            triangles[:, -3:] = f
            triangles[:, 0] = 3

            mesh = pv.PolyData(v, triangles, deep=False)
            return mesh

        def smooth_mesh(mesh, n_iter=100, **kwargs):
            """
            Adjust point coordinates using Laplacian smoothing.
            https://docs.pyvista.org/api/core/_autosummary/pyvista.PolyData.smooth.html#pyvista.PolyData.smooth

            Args:
                mesh: A mesh model.
                n_iter: Number of iterations for Laplacian smoothing.
                **kwargs: The rest of the parameters in pyvista.PolyData.smooth.

            Returns:
                smoothed_mesh: A smoothed mesh model.
            """

            smoothed_mesh = mesh.smooth(n_iter=n_iter, **kwargs)

            return smoothed_mesh

        uniform_surfs = []
        for sub_surf in mesh.split_bodies():
            sub_surf = unstruc_grid2polydata(sub_surf).triangulate().clean()
            # Get a smooth, uniformly meshed surface using voronoi clustering.
            sub_uniform_surf = self._uniform_re_mesh_single_fixed_voroni(
                mesh=sub_surf.extract_surface())  # sub_fix_surf.extract_surface()
            uniform_surfs.append(sub_uniform_surf)
        uniform_surf = self._merge_models(models=uniform_surfs)
        uniform_surf = uniform_surf.extract_surface().triangulate().clean()

        # post-process
        mesh = smooth_mesh(uniform_surf)
        return mesh

    @staticmethod
    def plt_3d(xyz, xyz_min, xyz_max):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        ax.set_xlim(xyz_min[0], xyz_max[0])
        ax.set_ylim(xyz_min[1], xyz_max[1])
        ax.set_zlim(xyz_min[2], xyz_max[2])
        plt.show()

    def find_repre_point(self, mesh_li, x_ran_sin=2.25):
        """
        # todo: 需要专利的话该方法可以写成专利

        Find the coordinate of the representing point of the 3D mesh of current cell type.

        We defined the representing point of a not-necessarily convex structure, as the result of iteratively and
        alternatively finding points that falls around the median position of orthogonal space, and picking up points
        that forms the biggest volume cluster. Specifically in our StereoSeq 3D system, for points in the mesh with
        the largest volume, we first find out its subset that falls 'onto' the median one or two z positions, cluster
        these points, find out points in the largest cluster, then again find out its subset that falls 'around' the
        median x position, re-cluster these points, find out points in the largest cluster, and eventually use its
        mean x, y and z coordinates, as the representing position of the structure.

        :param mesh_li: list of mesh, describing the same volume as self.scatter_li for each iterated element
        :param x_ran_sin: offset range of x coordinates when picking up the subset that falls 'around' the median x
                          position, described above.
        :return: coor_repre, NpArray of x, y and z coordinate of representing position.
        """

        def dic2arr(dic):
            return np.concatenate([np.expand_dims(dic['x'], axis=1),
                                   np.expand_dims(dic['y'], axis=1),
                                   np.expand_dims(dic['z'], axis=1)], axis=1)

        # (Preprocessing) 1. Pick up the mesh with the largest volume, in the mesh list
        vol_arr = np.array([mesh.volume for mesh in mesh_li])
        ind_max_vol = np.argmax(vol_arr)  # the index of the mesh with the largest volume

        # (Preprocessing) 2. Pick up the points from respective point cloud, that strictly falls in the above volume
        scatter = self.scatter_li[ind_max_vol]  # points used to generate the mesh
        xyz = dic2arr(scatter)
        points_po = pv.PolyData(xyz)
        points_po_sel = points_po.select_enclosed_points(mesh_li[ind_max_vol])
        xyz = points_po_sel.points
        # self.plt_3d(xyz, xyz_min, xyz_max)

        # 3. Pick up points falling on median slicing positions (z coordinate value) of above points
        li_sort_z = list(xyz[:, 2])
        li_sort_z.sort()
        # list of candidate z, considering spots along z are sampled with intervals
        z_med_li = list(
            set([li_sort_z[math.floor((len(li_sort_z) - 1) / 2)], li_sort_z[math.ceil((len(li_sort_z) - 1) / 2)]]))
        xyz_sel = xyz[np.isin(xyz[:, 2], z_med_li)]  # (n, 3)
        # self.plt_3d(xyz_sel, xyz_min, xyz_max)

        # 4. Find out the cluster of above points with the biggest approximate volume
        scatter_li = self._split_and_remove_outlier_append([], xyz_sel[:, 0], xyz_sel[:, 1], xyz_sel[:, 2],
                                                           self.ty_name, self.eps_val, self.min_samples,
                                                           self.thresh_num)
        len_arr = np.array([scatter['x'].shape[0] for scatter in scatter_li])
        ind_max_po = np.argmax(len_arr)
        scatter_max_clus = scatter_li[ind_max_po]  # {'x':, 'y':, 'z':, 'ty'}

        # 5. Pick up points whose x coordinate is not further than x_ran_sin from median x value of above points
        x_mean = np.median(scatter_max_clus['x'])
        xyz = dic2arr(scatter_max_clus)
        # self.plt_3d(xyz, xyz_min, xyz_max)
        xyz_sel = xyz[(xyz[:, 0] >= (x_mean - x_ran_sin)) & (xyz[:, 0] <= (x_mean + x_ran_sin))]  # (n, 3)
        assert xyz_sel.shape[0] >= 1, "No points fall into [x_mean - x_ran_sin, x_mean + x_ran_sin] range," \
                                      " try increasing x_ran_sin"

        # self.plt_3d(xyz_sel, xyz_min, xyz_max)
        # 6. Find out the cluster of above points with the biggest approximate volume
        scatter_li = self._split_and_remove_outlier_append([], xyz_sel[:, 0], xyz_sel[:, 1], xyz_sel[:, 2],
                                                           self.ty_name, self.eps_val, self.min_samples,
                                                           self.thresh_num)
        len_arr = np.array([scatter['x'].shape[0] for scatter in scatter_li])
        ind_max_po = np.argmax(len_arr)
        scatter_max_clus = scatter_li[ind_max_po]  # dict

        # 7. Use the central position of above points as representing point position
        xyz = dic2arr(scatter_max_clus)
        # self.plt_3d(xyz, xyz_min, xyz_max)
        coor_repre = np.mean(xyz, axis=0)

        return coor_repre

    def _create_mesh_alpha_shape(self, scatter, alpha):
        pcd = self._gen_o3d_pc(scatter)

        # if uniform_pc:
        #     pcd = self._uniform_larger_pc(pcd, type='o3d')

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

        if len(mesh.vertices) == 0:
            raise ValueError(
                f"The point cloud cannot generate a surface mesh with `alpha shape` method and alpha == {alpha}."
            )

        mesh = self._o3d_trimesh_to_pv_trimesh(mesh)
        return mesh

    def _create_mesh_ball_pivot(self, scatter, radii):
        pcd = self._gen_o3d_pc(scatter)
        # if uniform_pc:
        #     pcd = self._uniform_larger_pc(pcd, type='o3d')
        pcd.estimate_normals()
        radii = o3d.utility.DoubleVector(radii)

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)

        if len(mesh.vertices) == 0:
            raise ValueError(
                f"The point cloud cannot generate a surface mesh with `ball pivoting` method and radii == {radii}."
            )
        mesh = self._o3d_trimesh_to_pv_trimesh(mesh)
        return mesh

    def _create_mesh_poisson_surface(self, scatter, depth, width, scale, linear_fit, density_threshold):

        try:
            import open3d  # noqa
        except ImportError:
            raise ImportError("Need to install open3d")

        pcd = self._gen_o3d_pc(scatter)

        # if uniform_pc:
        #     pcd = self._uniform_larger_pc(pcd, type='o3d')

        pcd.estimate_normals()
        mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,
                                                                                  depth=depth, width=width, scale=scale,
                                                                                  linear_fit=linear_fit)
        if len(mesh.vertices) == 0:
            raise ValueError(
                f"The point cloud cannot generate a surface mesh with `poisson` method and depth == {depth}.")

        # A low density value means that the vertex is only supported by a low number of points from the input point cloud. # noqa
        if not (density_threshold is None):
            mesh.remove_vertices_by_mask(np.asarray(density) < np.quantile(density, density_threshold))

        mesh = self._o3d_trimesh_to_pv_trimesh(mesh)

        return mesh

    @staticmethod
    def _create_mesh_delaunay(scatter, alpha, tol):
        xyz = np.concatenate([np.expand_dims(scatter['x'], axis=1),
                              np.expand_dims(scatter['y'], axis=1),
                              np.expand_dims(scatter['z'], axis=1)], axis=1)

        pdata = pv.PolyData(xyz)
        # if uniform_pc:
        #     pcd = self._uniform_larger_pc(pcd, type='o3d')

        mesh = pdata.delaunay_3d(alpha=alpha).extract_surface().triangulate().clean(tolerance=tol)
        if mesh.n_points == 0:
            raise ValueError(
                f"\nThe point cloud cannot generate a surface mesh with `pyvista` method and alpha == {alpha}."
            )
        return mesh

    @staticmethod
    def _create_mesh_march_cubes_ori(scatter, num_pixel_short_edge, iso_val, rel_tol=0.01):
        # https://blog.csdn.net/u011426016/article/details/103140886
        # scale:
        import mcubes

        # prepare data: raw coordinates and transformation parameters
        xyz = np.concatenate([np.expand_dims(scatter['x'], axis=1),
                              np.expand_dims(scatter['y'], axis=1),
                              np.expand_dims(scatter['z'], axis=1)], axis=1)

        offset = np.min(xyz, axis=0)  # (3,)
        min_len = np.min(
            np.array([xyz[:, 0].max() - offset[0], xyz[:, 1].max() - offset[1], xyz[:, 2].max() - offset[2]]))
        scale = num_pixel_short_edge / min_len
        rigid_trans = {'offset': offset, 'scale': scale}

        # preprocess: transform coordinates according to transformation parameters
        xyz = xyz - rigid_trans['offset']
        xyz_scaled = np.ceil(rigid_trans['scale'] * xyz).astype(np.int64)

        # todo： 为什么要+3?
        vol = np.zeros(shape=(xyz_scaled[:, 0].max() + 3,
                              xyz_scaled[:, 1].max() + 3,
                              xyz_scaled[:, 2].max() + 3),
                       dtype=np.int64)
        vol[xyz_scaled[:, 0], xyz_scaled[:, 1], xyz_scaled[:, 2]] = 1

        # meshing using marching cubes
        vol = mcubes.smooth(vol)
        v, f = mcubes.marching_cubes(vol, iso_val)
        # print('v', v, v.shape)
        # print('f', f, f.shape)
        if len(v) == 0:
            raise ValueError("The point cloud cannot generate a surface mesh with `marching_cube` method.")

        # post_process: transform coordinates, datatypes
        v = np.asarray(v).astype(np.float64)  # 每一行表示一个节点的坐标
        v = v / rigid_trans['scale']
        v = v + rigid_trans['offset']

        f = np.asarray(f).astype(np.int64)
        f = np.c_[np.full(len(f), 3), f]  # 每一行表示一个面的组成点集合
        mesh = pv.PolyData(v, f.ravel()).extract_surface().triangulate().clean(tolerance=rel_tol, absolute=False)

        return mesh

    def _create_mesh_march_cubes(self, scatter, mc_scale_factor, levelset):

        def _scale_model_by_distance(
                model,
                distance=1,
                scale_center=None,
        ):

            # Check the distance.
            distance = distance if isinstance(distance, (tuple, list)) else [distance] * 3
            if len(distance) != 3:
                raise ValueError("`distance` value is wrong. \nWhen `distance` is a list or tuple, it can only"
                                 " contain three elements.")

            # Check the scaling center.
            scale_center = model.center if scale_center is None else scale_center
            if len(scale_center) != 3:
                raise ValueError("`scale_center` value is wrong." "\n`scale_center` can only contain three elements.")

            # Scale the model based on the distance.
            for i, (d, c) in enumerate(zip(distance, scale_center)):
                p2c_bool = np.asarray(model.points[:, i] - c) > 0
                model.points[:, i][p2c_bool] += d
                model.points[:, i][~p2c_bool] -= d

            return model

        def _scale_model_by_scale_factor(
                model,
                scale_factor=1,
                scale_center=None,
        ):

            # Check the scaling factor.
            scale_factor = scale_factor if isinstance(scale_factor, (tuple, list)) else [scale_factor] * 3
            if len(scale_factor) != 3:
                raise ValueError(
                    "`scale_factor` value is wrong."
                    "\nWhen `scale_factor` is a list or tuple, it can only contain three elements."
                )

            # Check the scaling center.
            scale_center = model.center if scale_center is None else scale_center
            if len(scale_center) != 3:
                raise ValueError("`scale_center` value is wrong." "\n`scale_center` can only contain three elements.")

            # Scale the model based on the scale center.
            for i, (f, c) in enumerate(zip(scale_factor, scale_center)):
                model.points[:, i] = (model.points[:, i] - c) * f + c

            return model

        def scale_model(
                model,
                distance=None,
                scale_factor=1,
                scale_center=None,
                inplace=False,
        ):
            """
            Scale the model around the center of the model.

            Args:
                model: A 3D reconstructed model.
                distance: The distance by which the model is scaled. If `distance` is float, the model is scaled same distance
                          along the xyz axis; when the `scale factor` is list, the model is scaled along the xyz axis at
                          different distance. If `distance` is None, there will be no scaling based on distance.
                scale_factor: The scale by which the model is scaled. If `scale factor` is float, the model is scaled along the
                              xyz axis at the same scale; when the `scale factor` is list, the model is scaled along the xyz
                              axis at different scales. If `scale_factor` is None, there will be no scaling based on scale factor.
                scale_center: Scaling center. If `scale factor` is None, the `scale_center` will default to the center of the model.
                inplace: Updates model in-place.

            Returns:
                model_s: The scaled model.
            """  # noqa

            model_s = model.copy() if not inplace else model

            if not (distance is None):
                model_s = _scale_model_by_distance(model=model_s, distance=distance, scale_center=scale_center)

            if not (scale_factor is None):
                model_s = _scale_model_by_scale_factor(model=model_s, scale_factor=scale_factor,
                                                       scale_center=scale_center)

            model_s = model_s.triangulate()

            return model_s if not inplace else None

        def rigid_transform(
                coords,
                coords_refA,
                coords_refB,
        ):
            """
            Compute optimal transformation based on the two sets of points and apply the transformation to other points.

            Args:
                coords: Coordinate matrix needed to be transformed.
                coords_refA: Referential coordinate matrix before transformation.
                coords_refB: Referential coordinate matrix after transformation.

            Returns:
                The coordinate matrix after transformation
            """
            # Check the spatial coordinates

            coords, coords_refA, coords_refB = coords.copy(), coords_refA.copy(), coords_refB.copy()
            assert (
                    coords.shape[1] == coords_refA.shape[1] == coords_refA.shape[1]
            ), "The dimensions of the input coordinates must be uniform, 2D or 3D."
            coords_dim = coords.shape[1]
            if coords_dim == 2:
                coords = np.c_[coords, np.zeros(shape=(coords.shape[0], 1))]
                coords_refA = np.c_[coords_refA, np.zeros(shape=(coords_refA.shape[0], 1))]
                coords_refB = np.c_[coords_refB, np.zeros(shape=(coords_refB.shape[0], 1))]

            # Compute optimal transformation based on the two sets of points.
            coords_refA = coords_refA.T
            coords_refB = coords_refB.T

            centroid_A = np.mean(coords_refA, axis=1).reshape(-1, 1)
            centroid_B = np.mean(coords_refB, axis=1).reshape(-1, 1)

            Am = coords_refA - centroid_A
            Bm = coords_refB - centroid_B
            H = Am @ np.transpose(Bm)

            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T

            if np.linalg.det(R) < 0:
                Vt[2, :] *= -1
                R = Vt.T @ U.T

            t = -R @ centroid_A + centroid_B

            # Apply the transformation to other points
            new_coords = (R @ coords.T) + t
            new_coords = np.asarray(new_coords.T)
            return new_coords[:, :2] if coords_dim == 2 else new_coords

        try:
            from scipy.spatial.distance import cdist
        except ImportError:
            raise ImportError('Need to install scipy')
        try:
            import mcubes
        except ImportError:
            raise ImportError('Need to install mcubes')

        # 1. data api: np.array to point cloud
        xyz = np.concatenate([np.expand_dims(scatter['x'], axis=1),
                              np.expand_dims(scatter['y'], axis=1),
                              np.expand_dims(scatter['z'], axis=1)], axis=1)
        pc = pv.PolyData(xyz)

        # if uniform_pc:
        #     pc = self._uniform_larger_pc(pc, type='pv')

        raw_points = np.asarray(pc.points)
        new_points = raw_points - np.min(raw_points, axis=0)
        pc.points = new_points

        # 2. Preprocess: calculate, then apply scale_factor to the points
        dist = cdist(XA=new_points, XB=new_points, metric="euclidean")
        row, col = np.diag_indices_from(dist)
        dist[row, col] = None
        max_dist = np.nanmin(dist, axis=1).max()  # the maximum neighbor-wise distance
        # so that mc_scale_factor times of maximum neighbor-wise distance equals to width of one voxel
        mc_sf = max_dist * mc_scale_factor

        scale_pc = scale_model(model=pc, scale_factor=1 / mc_sf)
        scale_pc_points = np.ceil(np.asarray(scale_pc.points)).astype(np.int64)
        scale_pc.points = scale_pc_points

        # 3. Preprocess: generate volume for mesh generation, based on the points
        volume_array = np.zeros(
            shape=[
                scale_pc_points[:, 0].max() + 3,
                scale_pc_points[:, 1].max() + 3,
                scale_pc_points[:, 2].max() + 3,
            ]
        )
        volume_array[scale_pc_points[:, 0], scale_pc_points[:, 1], scale_pc_points[:, 2]] = 1

        # 4. Process: extract the iso-surface based on marching cubes algorithm.
        volume_array = mcubes.smooth(volume_array)
        vertices, triangles = mcubes.marching_cubes(volume_array, levelset)
        if len(vertices) == 0:
            raise ValueError("The point cloud cannot generate a surface mesh with `marching_cube` method.")
        v = np.asarray(vertices).astype(np.float64)
        f = np.asarray(triangles).astype(np.int64)
        f = np.c_[np.full(len(f), 3), f]
        mesh = pv.PolyData(v, f.ravel()).extract_surface().triangulate()
        mesh.clean(inplace=True)

        # 5. Post-process: scale the mesh model back to the points' original coordinates
        mesh = scale_model(model=mesh, scale_factor=mc_sf)
        scale_pc = scale_model(model=scale_pc, scale_factor=mc_sf)
        mesh.points = rigid_transform(
            coords=np.asarray(mesh.points), coords_refA=np.asarray(scale_pc.points), coords_refB=raw_points
        )
        return mesh

    @staticmethod
    def _uniform_re_mesh_single_fixed_voroni(mesh, nsub=3, nclus=20000):
        """
        Generate a uniformly meshed surface using voronoi clustering.

        Args:
            mesh: A mesh model.
            nsub: Number of subdivisions. Each subdivision creates 4 new triangles, so the number of resulting
                triangles is nface*4**nsub where nface is the current number of faces.
            nclus: Number of voronoi clustering.

        Returns:
            new_mesh: A uniform mesh model.
        """

        # if mesh is not dense enough for uniform remeshing, increase the number of triangles in a mesh.
        if not (nsub is None):
            mesh.subdivide(nsub=nsub, subfilter="butterfly", inplace=True)

        # Uniformly remeshing.
        clustered = pyacvd.Clustering(mesh)

        if not (nsub is None):
            clustered.subdivide(nsub=nsub)

        clustered.cluster(nclus)

        new_mesh = clustered.create_mesh().triangulate().clean()
        return new_mesh

    # fixme: not adopted yet for its high consumption of time and memory
    def _uniform_larger_pc(self, pc, alpha=0, nsub=3, nclus=20000, type='pv'):
        """
        Generates a uniform point cloud with a larger number of points.
        If the number of points in the original point cloud is too small or the distribution of the original point
        cloud is not uniform, making it difficult to construct the surface, this method can be used for preprocessing.

        Args:
            pc: A point cloud model.
            alpha: Specify alpha (or distance) value to control output of this filter.
                   For a non-zero alpha value, only edges or triangles contained within a sphere centered at mesh vertices
                   will be output. Otherwise, only triangles will be output.
            nsub: Number of subdivisions. Each subdivision creates 4 new triangles, so the number of resulting triangles is
                  nface*4**nsub where nface is the current number of faces.
            nclus: Number of voronoi clustering.

        Returns:
            new_pc: A uniform point cloud with a larger number of points.
        """  # noqa
        coords = np.asarray(pc.points)
        coords_z = np.unique(coords[:, 2])

        slices = []
        for z in coords_z:
            slice_coords = coords[coords[:, 2] == z]
            slice_cloud = pv.PolyData(slice_coords)
            if len(slice_coords) >= 3:
                slice_plane = slice_cloud.delaunay_2d(alpha=alpha).triangulate().clean()
                uniform_plane = self._uniform_re_mesh_single_fixed_voroni(mesh=slice_plane, nsub=nsub, nclus=nclus)
                slices.append(uniform_plane)
            else:
                slices.append(slice_cloud)

        slices_mesh = self._merge_models(models=slices)
        if type in ['pv']:
            new_pc = pv.PolyData(slices_mesh.points).clean()
        elif type in ['o3d', 'open3d']:
            new_pc = o3d.geometry.PointCloud()
            new_pc.points = o3d.utility.Vector3dVector(slices_mesh.points)
        return new_pc

    def _remove_duplicated_mesh(self, mesh):
        """Removes unused points and degenerate cells."""
        """Remove replicated meshes, can accept multiple bodies in a mesh, named as 'clean_mesh' in Spateo"""

        sub_meshes = mesh.split_bodies()
        n_mesh = len(sub_meshes)

        if n_mesh == 1:
            return mesh
        else:
            inside_number = []
            for i, main_mesh in enumerate(sub_meshes[:-1]):
                main_mesh = pv.PolyData(main_mesh.points, main_mesh.cells)
                for j, check_mesh in enumerate(sub_meshes[i + 1:]):
                    check_mesh = pv.PolyData(check_mesh.points, check_mesh.cells)
                    inside = check_mesh.select_enclosed_points(main_mesh, check_surface=False).threshold(0.5)
                    inside = pv.PolyData(inside.points, inside.cells)
                    if check_mesh == inside:
                        inside_number.append(i + 1 + j)

            cm_number = list(set([i for i in range(n_mesh)]).difference(set(inside_number)))
            if len(cm_number) == 1:
                cmesh = sub_meshes[cm_number[0]]
            else:
                cmesh = self._merge_models([sub_meshes[i] for i in cm_number])

            return pv.PolyData(cmesh.points, cmesh.cells)

    def _fix_mesh(self, mesh):
        """
        Repair the mesh where it was extracted and subtle holes along complex parts of the mesh.
        Example in https://pymeshfix.pyvista.org/examples/index.html
        While filling subtles holes in mesh, this process may over fill some of the areas, hence
        produce overly coarse results in non-CAD meshes.

        """

        def fix_single_mesh(mesh):
            meshfix = mf.MeshFix(mesh)
            meshfix.repair(verbose=False)
            fixed_mesh = meshfix.mesh.triangulate().clean()

            if fixed_mesh.n_points == 0:
                raise ValueError("The surface cannot be Repaired. \nPlease change the method or parameters of "
                                 "surface reconstruction.")
            return fixed_mesh

        fixed_surfs = []
        for sub_surf in mesh.split_bodies():
            # Repair the surface mesh where it was extracted and subtle holes along complex parts of the mesh
            sub_fix_surf = fix_single_mesh(sub_surf.extract_surface())
            fixed_surfs.append(sub_fix_surf)
        fixed_surfs = self._merge_models(models=fixed_surfs)
        # fixed_surfs = fixed_surfs.extract_surface().triangulate().clean()
        return fixed_surfs

    @staticmethod
    def _merge_models(models):
        """Merge all models in the `models` list. The format of all models must be the same."""

        merged_model = models[0]
        for model in models[1:]:
            merged_model = merged_model.merge(model)

        return merged_model

    @staticmethod
    def _split_and_remove_outlier_append(scatter_li, x_arr, y_arr, z_arr, ty,
                                         eps_val, min_samples, thresh_num):

        X = np.concatenate([np.expand_dims(x_arr, axis=1),
                            np.expand_dims(y_arr, axis=1),
                            np.expand_dims(z_arr, axis=1)], axis=1)  # (n,3)

        # 2. re-grouping based on spatial coordinates
        dbscan_clus = DBSCAN(eps=eps_val, min_samples=min_samples).fit(X)
        dbscan_labels_arr = dbscan_clus.labels_  # (n_spot_in_this_cluster,)

        # 3. process data into input required by volume visualization, based on re-grouping results
        for dbscan_label in set(dbscan_labels_arr):
            # get rid of outliers
            if dbscan_label == -1:
                continue
            grp_idx = np.where(dbscan_labels_arr == dbscan_label)
            # print(dbscan_label, grp_idx[0].shape)
            if grp_idx[0].shape[0] < thresh_num:
                continue
            sin_vol = {'x': x_arr[grp_idx], 'y': y_arr[grp_idx], 'z': z_arr[grp_idx], 'ty': ty}
            scatter_li.append(sin_vol)

        return scatter_li

    def _select_split_and_remove_outlier(self, eps_val, min_samples, thresh_num=10):
        """
        Generate inner volumes of the selected type to be visualized.
        :param ty_name: List of types
        :param xli: List of x, with the same order of cells / bins as other iterable inputs
        :param yli: List of y, with the same order of cells / bins as other iterable inputs
        :param zli: List of z, with the same order of cells / bins as other iterable inputs
        :param tyli: List of type, with the same order of cells / bins as other iterable inputs
        :param eps_val: In DBSCAN, the maximum distance between two samples for one to be considered as in the neighborhood
                        of the other. Try increase this value and min_samples if a cell type region is splitted into too
                        many subregions.
        :param min_samples: In DBSCAN, the number of samples (or total weight) in a neighborhood for a point to be
                            considered as a core point. This includes the point itself. Try increase this value and
                             eps_val if a cell type region is splitted into too many subregions. Try increase this value if
                             too many cells are excluded as outliers.
        :param thresh_num: minimum number of cells in a group to be considered to display a mesh
        :return: List of dictionary of inner volumes. Each dictionary includes keys of 'x', 'y', 'z', with values in
                NumPy.NdArray, and 'ty' with values of datatype str.
        """  # noqa

        # scatter_li = []
        # for ty in list(dict.fromkeys(self.tyli)):
        #     if not ty == self.ty_name:
        #         continue
        #
        #     # print(ty)

        # 1. prepare data for analysis and processing
        ty_idx = np.where(np.array(self.tyli) == self.ty_name)

        x_arr = np.array(self.xli)[ty_idx]  # (n,)
        y_arr = np.array(self.yli)[ty_idx]
        z_arr = np.array(self.zli)[ty_idx]
        ty_arr = np.array(self.tyli)[ty_idx]

        scatter_li = self._split_and_remove_outlier_append([], x_arr, y_arr, z_arr, ty_arr[0], eps_val=eps_val,
                                                           min_samples=min_samples, thresh_num=thresh_num)
        return scatter_li

    @staticmethod
    def _o3d_trimesh_to_pv_trimesh(trimesh):
        """Convert a triangle mesh in Open3D to PyVista."""
        v = np.asarray(trimesh.vertices)
        f = np.array(trimesh.triangles)
        f = np.c_[np.full(len(f), 3), f]

        mesh = pv.PolyData(v, f.ravel()).extract_surface().triangulate().clean()
        return mesh

    @staticmethod
    def _gen_o3d_pc(scatter):
        xyz = np.concatenate([np.expand_dims(scatter['x'], axis=1),
                              np.expand_dims(scatter['y'], axis=1),
                              np.expand_dims(scatter['z'], axis=1)], axis=1)
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(xyz)
        return o3d_pcd


class GenMesh(AlgorithmBase):
    def main(
            self,
            cluster_res_key=None,
            ty_name_li=None,
            method='march',
            eps_val=2,
            min_samples=5,
            thresh_num=10,
            key_name='mesh',
            alpha=None, radii=None, depth=None, width=None, scale=None, linear_fit=None, density_threshold=None,
            mc_scale_factor=None, levelset=None, tol=None):
        # todo: 在 __init__中暴露出去
        """

        :param stereo_exp_data: stereo_exp_data to be add mesh result on
        :param cluster_res_key: the key which specifies the clustering result in data.tl.result.
        :param ty_name_li: list of cell types to be computed mesh
        :param method: 'march', 'march_cubes'
                Method parameter among 'march' and 'march_cubes' indicates using Marching Cubes Algorithm to create mesh. Marching Cubes
                voxelize a point cloud and assigns 0 / 1 to each voxel/cube, then finds the iso face of each cube that
                intersects the edges of the cube into opposite classification. A mesh is formed by the iso faces.

                It is robust to noises and lackage of points, but may add extra mesh that doesn't exist physically.

                    We recommend you to browse https://www.cs.carleton.edu/cs_comps/0405/shape/marching_cubes.html for visualization
                    of this algorithm. Refer to its publication for more details: https://dl.acm.org/doi/pdf/10.1145/37402.37422

                    -------------------
                    'alpha', 'alpha_shape'

                    Method parameter among 'alpha' and 'alpha_shape' indicates using Alpha Shapes Algorithm to create mesh. Alpha
                    Shapes specifies mesh of point cloud, by finding all node-node combinations that supports the sphere to
                    pass them without surrounding a third point.

                    Though capable of generating both concave and convex surfaces, it is not robust enough when dealing with
                    non-uniformly distributed points, or relatively small hollows, since its alpha value is globally adopted,
                    instead of locally optimized.

                    We recommend you to read
                    https://graphics.stanford.edu/courses/cs268-11-spring/handouts/AlphaShapes/as_fisher.pdf
                    for a more in-depth understanding of this algorithm, or to read its publication for more details:
                    https://www.cs.jhu.edu/~misha/Fall05/Papers/edelsbrunner94.pdf

                    --------------------
                    'ball', 'ball_pivot'

                    Method parameter value among 'ball' and 'ball_pivot' indicates using Ball Pivoting Algorithm (BPA) to create mesh.
                    BPA is quite closed to Alpha in terms of its process. In BPA, Three points form a triangle if a ball of
                    certain radius touches them without containing any other point. Starting with a seed triangle, the ball
                    pivots around an edge until it touches another point, forming another triangle. The process continues
                    until all reachable edges have been tried, and then starts from another seed triangle, until all points
                    are considered.

                    Though capable of generating both concave and convex surfaces, it is not robust enough when dealing with
                    non-uniformly distributed points, or relatively small hollows, since its alpha value is globally adopted,
                    instead of locally optimized. We have also noticed some failures in mesh generation due to points adopted
                    for initialization.

                    Refer to its publication for more details
                    https://vgc.poly.edu/~csilva/papers/tvcg99.pdf

                    ---------------------
                    'poisson', 'poisson_surface'

                    Method parameter value among 'poisson' and 'poisson_surface' indicates using Poisson Mesh Reconstruction
                    to create mesh. It divides the spatial domain by octrees, uses the sub-domains to construct the point domain
                    in different ways (called indicator function), then solves the Poisson problem equation of divergence
                    of gradient of indicator function equals to divergence of oriented points. This poisson equation, which is
                    naturally a Partial Differential Equation, can be then transformed to a Optimization problem without restrictions
                    for solution.

                    Poisson Surface provides benefits of noise robustness due to its global algorithm nature, compared to other
                    mesh reconstruction methods. However, cases with non-uniformly distributed points, e.g. cases with higher slice interval
                    than spot intervals, poorly annotated / clustered, etc., may cast huge side effect to its performance.

                    Refer to its publication for more details: https://hhoppe.com/poissonrecon.pdf

                    -------------------------
                    'delaunay', 'delaunay_3d'

                    Method parameter value among 'delaunay' and 'delaunay_3d' indicates using Delaunay 3d Mesh Reconstruction
                    to create mesh. It iteratively finds out the 3D triangulation of points with tedrahedra close
                    to regular tedrahedra, using Bowyer-Watson algorithm, then extracts the surface to form a mesh. The
                    output of the Delaunay triangulation is supposedly a convex hull, hence may surafce that 'overly wrap'
                    points clouds, which idealy forms concave shapes.

                    It is most idealized to be used in situations where the convex shape is known as prior knowledge.

                    Refer to its publication for more details:
                    https://www.kiv.zcu.cz/site/documents/verejne/vyzkum/publikace/technicke-zpravy/2002/tr-2002-02.pdf
        :param eps_val:  In DBSCAN, the maximum distance between two samples for one to be considered as in the neighborhood
                            of the other. Try increase this value and min_samples if a cell type region is splitted into too
                            many subregions.
        :param min_samples:  In DBSCAN, the number of samples (or total weight) in a neighborhood for a point to be
                            considered as a core point. This includes the point itself. Try increase this value and
                            eps_val if a cell type region is splitted into too many subregions. Try increase this value if
                            too many cells are excluded as outliers.
        :param thresh_num: minimum number of cells in a group to be considered to display a mesh
        :param key_name: name of key in the dictionary of adata.ubs['mesh'], mesh algorithm name and its input parameter value
                        are recommended as key_name, e.g. 'delaunay_3d_tol_0.01'
        :param alpha: If method is among 'alpha' and 'alpha_shape', it is the radius of sphere that is used to construct
                        mesh. Or else if alpha is among 'delaunay' and 'delaunay_3d', alpha is the distance value to control
                        output of this filter. For a non-zero alpha value, only vertices, edges, faces, or tetrahedra
                        contained within the circumsphere (of radius alpha) will be output. Otherwise, only tetrahedra will
                        be output.
        :param radii: If method is among 'ball' and 'ball_pivot', it is the radii of the ball that are used for the
                        surface reconstruction.
        :param depth:  If method is among 'poisson' and 'poisson_surface', it is the maximum depth of the tree that will
                        be used for surface reconstruction.
        :param width: If method is among 'poisson' and 'poisson_surface', it specifies the target width of the finest
                        level octree cells.
        :param scale: If method is among 'poisson' and 'poisson_surface', it specifies the ratio between the diameter of
                        the cube used for reconstruction and the diameter of the samples' bounding cube.
        :param linear_fit: If method is among 'poisson' and 'poisson_surface', if true, the reconstructor will use linear
                            interpolation to estimate the positions of iso-vertices.
        :param density_threshold: If method is among 'poisson' and 'poisson_surface', the mesh with density lower than
                                    this quantile will be filtered.
        :param mc_scale_factor:  scale_factor adpoted in the Marching Cubes method, so that mc_scale_factor times of
                                maximum neighbor-wise distance equals to width of one voxel
        :param levelset: If method in 'march' and 'march_cubes', this is the iso value when generating the iso surfaces.
        :param tol: If method in 'delaunay', 'delaunay_3d', cells smaller than this will be degenerated and merged.

        :return: adata, mesh written in stereo_exp_data.tl.result['mesh']
        """  # noqa

        def scatter2xyz(scatter):
            xyz = np.concatenate([np.expand_dims(scatter['x'], axis=1),
                                  np.expand_dims(scatter['y'], axis=1),
                                  np.expand_dims(scatter['z'], axis=1)], axis=1)
            return xyz

        stereo_exp_data = self.stereo_exp_data

        xli = stereo_exp_data.position[:, 0].tolist()
        yli = stereo_exp_data.position[:, 1].tolist()
        zli = stereo_exp_data.position_z.reshape(-1).tolist()
        if cluster_res_key in stereo_exp_data.cells._obs.columns:
            tyli = stereo_exp_data.cells._obs[cluster_res_key].tolist()
        else:
            tyli = self.pipeline_res[cluster_res_key]['group'].tolist()

        if not 'mesh' in self.pipeline_res:  # noqa
            self.pipeline_res['mesh'] = {}
        self.pipeline_res['mesh'][key_name] = {}

        if ty_name_li is None:
            ty_name_li = list(dict.fromkeys(tyli).keys())
        else:
            ty_name_li = list(dict.fromkeys(ty_name_li).keys())  # remove duplicates just in case
        for ty_name in ty_name_li:
            try:
                tdg = ThreeDimGroup(xli, yli, zli, tyli, ty_name=ty_name, eps_val=eps_val, min_samples=min_samples,
                                    thresh_num=thresh_num)  # 1.5, 8
                # print([(ele['ty'], ele['x'].shape[0]) for ele in tdg.scatter_li])
                # ty_set = set([dic['ty'] for dic in mesh.scatter_li])
                # ty_id = dict(zip(ty_set, range(len(ty_set))))
                # print(ty_id)

                mesh, mesh_li = tdg.create_mesh_of_type(method=method, alpha=alpha, radii=radii,
                                                        depth=depth, width=width, scale=scale, linear_fit=linear_fit,
                                                        density_threshold=density_threshold,
                                                        mc_scale_factor=mc_scale_factor, levelset=levelset,
                                                        tol=tol)
                # mesh.plot()
                # print(mesh)

                # print(tdg.find_repre_point(mesh_li, x_ran_sin=2.25))
                # mesh = tdg.create_mesh_of_type('CNS', method='alpha', alpha=10)
                # mesh = tdg.create_mesh_of_type('CNS', method='ball', radii=[50])
                # mesh = tdg.create_mesh_of_type('CNS', method='poisson', depth=5, scale=2.5, density_threshold=0.3)
                # mesh = tdg.create_mesh_of_type('all', method='delaunay', tol=0.001)  # alpha
                # mesh.plot()

                # pv.plot(scatter2xyz(tdg.scatter_li[0]))

                mesh = tdg.uniform_re_mesh_and_smooth(mesh)
                # print(mesh)
                # mesh.plot()

                # _ = pl.add_mesh(mesh)

                # tdg = ThreeDimGroup(xli, yli, zli, tyli, ty_name='all', eps_val=2, min_samples=5, thresh_num=10)  # 1.5, 8 # noqa
                # mesh, mesh_li = tdg.create_mesh_of_type(method='march', mc_scale_factor=1.5)
                # mesh = tdg.uniform_re_mesh_and_smooth(mesh)
                # _ = pl.add_mesh(mesh)
                # pl.export_obj('E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paste_based/data/fruitfly_embryo/bin_recons_spot_level/scene.obj') # noqa

                # mesh.save()

                # this is to show polydata can be accessed by plotly
                # import plotly.graph_objects as go
                # import plotly.io as pio
                # pio.renderers.default = "browser"
                #
                # points = mesh.points
                # triangles = mesh.faces.reshape(-1, 4)
                # fig = go.Figure((
                #                 go.Mesh3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                #                            i=triangles[:, 1], j=triangles[:, 2], k=triangles[:, 3]),
                #                  go.Scatter3d(x=scatter['x'], y=scatter['y'], z=scatter['z'], marker=dict(size=2))))
                # fig.show()
                self.pipeline_res['mesh'][key_name][ty_name] = {}
                self.pipeline_res['mesh'][key_name][ty_name]['points'] = np.ndarray(shape=mesh.points.shape,
                                                                                    dtype=mesh.points.dtype,
                                                                                    buffer=mesh.points)
                mfaces = mesh.faces.reshape(-1, 4)
                self.pipeline_res['mesh'][key_name][ty_name]['faces'] = np.ndarray(shape=mfaces.shape,
                                                                                   dtype=mfaces.dtype,
                                                                                   buffer=mfaces)
            except Exception as e:
                print(e)

        return stereo_exp_data
