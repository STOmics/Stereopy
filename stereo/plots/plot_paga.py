import random
from types import MappingProxyType
from typing import (
    Optional,
    Union,
    Mapping,
    Any,
    Literal
)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from stereo.log_manager import logger
from stereo.plots.plot_base import PlotBase

_Layout = Literal[
    'fr',
    'drl',
    'kk',
    'grid_fr',
    'lgl',
    'rt',
    'rt_circular',
    'fa'
]


class PlotPaga(PlotBase):

    def _get_igraph_from_adjacency(self, adjacency, directed=None):
        """
        Get igraph graph from adjacency matrix.
        param adjacency: a sparse adjacency matrix of a graph
        return: igraph Graph object
        """
        import igraph as ig

        sources, targets = adjacency.nonzero()
        weights = adjacency[sources, targets]
        if isinstance(weights, np.matrix):
            weights = weights.A1
        g = ig.Graph(directed=directed)
        g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
        g.add_edges(list(zip(sources, targets)))
        try:
            g.es['weight'] = weights
        except KeyError:
            pass
        if g.vcount() != adjacency.shape[0]:
            logger.warning(
                f'The constructed graph has only {g.vcount()} nodes. '
                'Your adjacency matrix contained redundant nodes.'
            )
        return g

    def _compute_pos(
            self,
            adjacency,
            layout=None,
            random_state=0,
            init_pos=None,
            adj_tree=None,
            root=0,
            layout_kwds: Mapping[str, Any] = MappingProxyType({}),
    ):
        """
        compute the position for each node
        param adjacency: adjacent matrix of a graph
        param layout: the method to layout each node
        param random_state: to control the random initialization
        param adj_tree: required for some layout method
        return: a 2-dim array with x-y position for each node
        """
        import random
        import networkx as nx

        random_state = check_random_state(random_state)
        nx_g_solid = nx.Graph(adjacency)
        if layout is None:
            layout = 'fr'
        if layout == 'fa':
            try:
                from fa2 import ForceAtlas2
            except ImportError:
                logger.warning(
                    "Package 'fa2' is not installed, falling back to layout 'fr'."
                    'To use the faster and better ForceAtlas2 layout, '
                    "install package 'fa2' (`pip install fa2`)."
                )
                layout = 'fr'
        if layout == 'fa':
            if init_pos is None:
                init_coords = random_state.random_sample((adjacency.shape[0], 2))
            else:
                init_coords = init_pos.copy()
            forceatlas2 = ForceAtlas2(
                # Behavior alternatives
                outboundAttractionDistribution=False,  # Dissuade hubs
                linLogMode=False,  # NOT IMPLEMENTED
                adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                edgeWeightInfluence=1.0,
                # Performance
                jitterTolerance=1.0,  # Tolerance
                barnesHutOptimize=True,
                barnesHutTheta=1.2,
                multiThreaded=False,  # NOT IMPLEMENTED
                # Tuning
                scalingRatio=2.0,
                strongGravityMode=False,
                gravity=1.0,
                # Log
                verbose=False,
            )
            if 'maxiter' in layout_kwds:
                iterations = layout_kwds['maxiter']
            elif 'iterations' in layout_kwds:
                iterations = layout_kwds['iterations']
            else:
                iterations = 500
            pos_list = forceatlas2.forceatlas2(
                adjacency, pos=init_coords, iterations=iterations
            )
            pos = {n: [p[0], -p[1]] for n, p in enumerate(pos_list)}
        else:
            # igraph layouts
            random.seed(random_state.bytes(8))
            g = self._get_igraph_from_adjacency(adjacency)
            if 'rt' in layout:
                g_tree = self._get_igraph_from_adjacency(adj_tree)
                pos_list = g_tree.layout(
                    layout, root=root if isinstance(root, list) else [root]
                ).coords
            elif layout == 'circle':
                pos_list = g.layout(layout).coords
            else:
                # I don't know why this is necessary
                if init_pos is None:
                    init_coords = random_state.random_sample((adjacency.shape[0], 2)).tolist()
                else:
                    init_pos = init_pos.copy()
                    # this is a super-weird hack that is necessary as igraph’s
                    # layout function seems to do some strange stuff here
                    init_pos[:, 1] *= -1
                    init_coords = init_pos.tolist()
                try:
                    pos_list = g.layout(
                        layout, seed=init_coords, weights='weight', **layout_kwds
                    ).coords
                except AttributeError:  # hack for empty graphs...
                    pos_list = g.layout(layout, seed=init_coords, **layout_kwds).coords
            pos = {n: [p[0], -p[1]] for n, p in enumerate(pos_list)}
        if len(pos) == 1:
            pos[0] = (0.5, 0.5)
        pos_array = np.array([pos[n] for count, n in enumerate(nx_g_solid)])
        return pos_array

    def paga_plot(
            self,
            adjacency: str = 'connectivities_tree',
            threshold: float = 0.01,
            layout: _Layout = 'fr',
            random_state: int = 0,
            cmap: str = 'tab20',
            ax: Optional[plt.Axes] = None,
            width: Optional[int] = None,
            height: Optional[int] = None,
            dot_size: Optional[int] = 30
    ):
        """
        abstract paga plot for the paga result.

        :param adjacency: keyword to use for paga or paga tree, available values include 'connectivities' and 'connectivities_tree'. # noqa
        :param threshold: prune edges lower than threshold.
        :param layout: the method to layout each node.
        :param random_state: to control the random initializatio.
        :param cmap: colormap to use, default with tab20.
        :param ax: subplot to plot.
        :param width: the figure width.
        :param height: the figure height.
        :param dot_size: The marker size in points**2 (typographic points are 1/72 in.).
            Default is 30.

        """
        # calculate node positions
        adjacency_mat = self.pipeline_res['paga'][adjacency].copy()
        if threshold > 0:
            adjacency_mat.data[adjacency_mat.data < threshold] = 0
            adjacency_mat.eliminate_zeros()

        pos = self._compute_pos(adjacency_mat, layout=layout, random_state=random_state)
        self.pipeline_res['paga']['pos'] = pos

        # network
        G = pd.DataFrame(adjacency_mat.todense())
        ct_list = self.stereo_exp_data.cells[self.pipeline_res['paga']['groups']].cat.categories
        G.index = ct_list
        G.columns = ct_list
        Edges = nx.from_pandas_adjacency(G).edges()
        Nodes2pos = dict(zip(ct_list, list(pos)))

        # define colors
        color_list = plt.get_cmap(cmap, len(ct_list))

        # plotting
        if ax is None:
            _, ax = plt.subplots(1)
        if width is not None:
            ax.get_figure().set_figwidth(width)
        if height is not None:
            ax.get_figure().set_figheight(height)

        ax.scatter(pos[:, 0], pos[:, 1], c=color_list.colors, zorder=1, s=dot_size)
        for i in range(len(ct_list)):
            ax.text(pos[i, 0], pos[i, 1], s=ct_list[i], zorder=2)
        for i, j in Edges:
            xi, yi = Nodes2pos[i]
            xj, yj = Nodes2pos[j]
            ax.plot([xi, xj], [yi, yj], color='black', zorder=0)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax.get_figure()

    def _draw_graph(
            self,
            layout: _Layout = 'fa',
            init_pos: Union[str, bool, None] = None,
            root: Optional[int] = None,
            random_state=0,
            n_jobs: Optional[int] = None,
            adjacency=None,
            key_added_ext: Optional[str] = None,
            obsp: Optional[str] = None,
            copy: bool = False,
            **kwds,
    ):
        """
        Force-directed graph drawing [Islam11]_ [Jacomy14]_ [Chippada18]_.

        An alternative to tSNE that often preserves the topology of the data
        better. This requires to run :func:`~scanpy.pp.neighbors`, first.

        The default layout ('fa', `ForceAtlas2`) [Jacomy14]_ uses the package |fa2|_
        [Chippada18]_, which can be installed via `pip install fa2`.

        `Force-directed graph drawing`_ describes a class of long-established
        algorithms for visualizing graphs.
        It has been suggested for visualizing single-cell data by [Islam11]_.
        Many other layouts as implemented in igraph [Csardi06]_ are available.
        Similar approaches have been used by [Zunder15]_ or [Weinreb17]_.

        .. |fa2| replace:: `fa2`
        .. _fa2: https://github.com/bhargavchippada/forceatlas2
        .. _Force-directed graph drawing: https://en.wikipedia.org/wiki/Force-directed_graph_drawing

        Parameters
        ----------
        adata
            Annotated data matrix.
        layout
            'fa' (`ForceAtlas2`) or any valid `igraph layout
            <http://igraph.org/c/doc/igraph-Layout.html>`__. Of particular interest
            are 'fr' (Fruchterman Reingold), 'grid_fr' (Grid Fruchterman Reingold,
            faster than 'fr'), 'kk' (Kamadi Kawai', slower than 'fr'), 'lgl' (Large
            Graph, very fast), 'drl' (Distributed Recursive Layout, pretty fast) and
            'rt' (Reingold Tilford tree layout).
        root
            Root for tree layouts.
        random_state
            For layouts with random initialization like 'fr', change this to use
            different intial states for the optimization. If `None`, no seed is set.
        adjacency
            Sparse adjacency matrix of the graph, defaults to neighbors connectivities.
        key_added_ext
            By default, append `layout`.
        proceed
            Continue computation, starting off with 'X_draw_graph_`layout`'.
        init_pos
            `'paga'`/`True`, `None`/`False`, or any valid 2d-`.obsm` key.
            Use precomputed coordinates for initialization.
            If `False`/`None` (the default), initialize randomly.
        neighbors_key
            If not specified, draw_graph looks .obsp['connectivities'] for connectivities
            (default storage place for pp.neighbors).
            If specified, draw_graph looks
            .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.
        obsp
            Use .obsp[obsp] as adjacency. You can't specify both
            `obsp` and `neighbors_key` at the same time.
        copy
            Return a copy instead of writing to adata.
        **kwds
            Parameters of chosen igraph layout. See e.g. `fruchterman-reingold`_
            [Fruchterman91]_. One of the most important ones is `maxiter`.

            .. _fruchterman-reingold: http://igraph.org/python/doc/igraph.Graph-class.html#layout_fruchterman_reingold

        Returns
        -------
        Depending on `copy`, returns or updates `adata` with the following field.

        **X_draw_graph_layout** : `adata.obsm`
            Coordinates of graph layout. E.g. for layout='fa' (the default),
            the field is called 'X_draw_graph_fa'
        """
        adjacency = self.pipeline_res['neighbors']['connectivities']
        # init coordinates
        groups_key = self.pipeline_res['paga']['groups']
        groups = self.stereo_exp_data.cells.to_df()[groups_key]

        pos = self.pipeline_res['paga']['pos']
        connectivities_coarse = self.pipeline_res['paga']['connectivities_tree'].copy()

        init_pos = np.ones((adjacency.shape[0], 2))
        for i, group_pos in enumerate(pos):
            subset = (groups == groups.cat.categories[i]).values
            neighbors = connectivities_coarse[i].nonzero()
            if len(neighbors[1]) > 0:
                connectivities = connectivities_coarse[i][neighbors]
                nearest_neighbor = neighbors[1][np.argmax(connectivities)]
                noise = np.random.random((len(subset[subset]), 2))
                dist = pos[i] - pos[nearest_neighbor]
                noise = noise * dist
                init_pos[subset] = group_pos - 0.5 * dist + noise
            else:
                init_pos[subset] = group_pos

        # see whether fa2 is installed
        if layout == 'fa':
            try:
                from fa2 import ForceAtlas2
            except ImportError:
                logger.warning(
                    "Package 'fa2' is not installed, falling back to layout 'fr'."
                    'To use the faster and better ForceAtlas2 layout, '
                    "install package 'fa2' (`pip install fa2`)."
                )
                layout = 'fr'
        # actual drawing
        if layout == 'fa':
            forceatlas2 = ForceAtlas2(
                # Behavior alternatives
                outboundAttractionDistribution=False,  # Dissuade hubs
                linLogMode=False,  # NOT IMPLEMENTED
                adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                edgeWeightInfluence=1.0,
                # Performance
                jitterTolerance=1.0,  # Tolerance
                barnesHutOptimize=True,
                barnesHutTheta=1.2,
                multiThreaded=False,  # NOT IMPLEMENTED
                # Tuning
                scalingRatio=2.0,
                strongGravityMode=False,
                gravity=1.0,
                # Log
                verbose=False,
            )
            if 'maxiter' in kwds:
                iterations = kwds['maxiter']
            elif 'iterations' in kwds:
                iterations = kwds['iterations']
            else:
                iterations = 500
            positions = forceatlas2.forceatlas2(
                adjacency, pos=init_pos, iterations=iterations
            )
            positions = np.array(positions)
        else:
            # igraph doesn't use numpy seed
            random.seed(random_state)

            g = self._get_igraph_from_adjacency(adjacency)
            if layout in {'fr', 'drl', 'kk', 'grid_fr'}:
                ig_layout = g.layout(layout, seed=init_pos.tolist(), **kwds)
            elif 'rt' in layout:
                if root is not None:
                    root = [root]
                ig_layout = g.layout(layout, root=root, **kwds)
            else:
                ig_layout = g.layout(layout, **kwds)
            positions = np.array(ig_layout.coords)
        self.stereo_exp_data.cells_matrix['paga_pos'] = positions
        return positions

    def draw_graph(
            self,
            adjacency: str = 'connectivities_tree',
            color: Optional[str] = None,
            size: int = 1,
            threshold: float = 0.01,
            layout: _Layout = 'fr',
            random_state: int = 0,
            cmap: str = 'tab20',
            width: int = 15,
            height: int = 6,
            dot_size: int = 30
    ):
        """
        Force-directed graph drawing

        :param adjacency: keyword to use for paga or paga tree, available values include 'connectivities' and 'connectivities_tree'. # noqa
        :param color: the col in cells or a gene name to display in compare plot.
        :param size: cell spot size.
        :param threshold: prune edges lower than threshold.
        :param layout: the method to layout each node.
        :param random_state: to control the random initialization.
        :param cmap: colormap to use, default with tab20.
        :param width: the figure width.
        :param height: the figure height.
        :param dot_size: The marker size in points**2 (typographic points are 1/72 in.).
            Default is 30.

        """
        # parameter setting
        fig = plt.figure(figsize=(width, height))
        ax = plt.subplot(1, 2, 1)
        # network
        self.paga_plot(adjacency=adjacency, threshold=threshold, random_state=random_state, cmap=cmap, ax=ax,
                       dot_size=dot_size)

        # cell position
        cell_pos = self._draw_graph(layout=layout)

        # plotting
        ax = plt.subplot(1, 2, 2)
        if color is None:
            color = self.pipeline_res['paga']['groups']
        if (color not in self.stereo_exp_data.cells) and (color not in self.stereo_exp_data.gene_names):
            logger.info(
                f"color is neither in cells nor in genes, use '{self.pipeline_res['paga']['groups']}' as default")
            color = self.pipeline_res['paga']['groups']

        if color in self.stereo_exp_data.cells:
            if self.stereo_exp_data.cells[color].dtype == 'category':
                ax.scatter(cell_pos[:, 0], cell_pos[:, 1], s=size,
                           c=self.stereo_exp_data.cells[color].cat.codes.to_numpy(), cmap=cmap)
                cell_pos_df = pd.DataFrame(cell_pos)
                cell_pos_df[color] = self.stereo_exp_data.cells[color].to_list()
                cell_center_pos_df = cell_pos_df.groupby(color).mean()
                for s, row in cell_center_pos_df.iterrows():
                    ax.text(row[0], row[1], s)
            else:
                ax.scatter(cell_pos[:, 0], cell_pos[:, 1], s=size, c=self.stereo_exp_data.cells[color])
        elif color in self.stereo_exp_data.gene_names:
            gene_list = list(self.stereo_exp_data.genes.to_df().index)
            gene_index = gene_list.index(color)
            clist = self.stereo_exp_data.exp_matrix[:, gene_index]
            ax.scatter(cell_pos[:, 0], cell_pos[:, 1], s=size, c=clist)

        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    def paga_compare(
            self,
            basis: Optional[str] = None,
            adjacency: str = 'connectivities_tree',
            color: Optional[str] = None,
            size: int = 1,
            threshold: float = 0.01,
            cmap: str = 'tab20',
            width: int = 15,
            height: int = 6,
            dot_size: int = 30
    ):
        """
        abstract paga plot for the paga result and cell distribute around paga.

        :param basis: the embedding of cells, default
        :param adjacency: keyword to use for paga or paga tree, available values include 'connectivities' and 'connectivities_tree'. # noqa
        :param color: the col in cells or a gene name to display in compare plot.
        :param size: cell spot size.
        :param threshold: prune edges lower than threshold.
        :param cmap: colormap to use, default with tab20.
        :param width: the figure width.
        :param height: the figure height.
        :param dot_size: The marker size in points**2 (typographic points are 1/72 in.).
            Default is 30.

        """

        if color is None:
            color = self.pipeline_res['paga']['groups']
        if (color not in self.stereo_exp_data.cells) and (color not in self.stereo_exp_data.gene_names):
            logger.info(
                f"color is neither in cells nor in genes, use '{self.pipeline_res['paga']['groups']}' as default")
            color = self.pipeline_res['paga']['groups']

        if basis is None:
            basis = 'umap'
        if basis not in self.pipeline_res:
            logger.info(
                f"{basis} is not in result, use 'umap' as default")
            basis = 'umap'
            if basis not in self.pipeline_res:
                logger.info(
                    f"umap is not in result, please run umap first, try to use 'pca' instead")
                basis = 'pca'
                if basis not in self.pipeline_res:
                    logger.info(
                        f"pca is not in result, please specify a dim reduction result, try to run pca or umap first.")
                    raise KeyError("basis not found: " + str(basis))

        df = self.pipeline_res[basis].copy()
        df = df[[0, 1]]
        df[color] = list(self.stereo_exp_data.cells[color])
        pos = df.groupby(color).mean()

        # calculate node positions
        adjacency_mat = self.pipeline_res['paga'][adjacency].copy()
        if threshold > 0:
            adjacency_mat.data[adjacency_mat.data < threshold] = 0
            adjacency_mat.eliminate_zeros()

        # network
        G = pd.DataFrame(adjacency_mat.todense())
        ct_list = self.stereo_exp_data.cells[color].cat.categories

        pos = pos.loc[ct_list]
        pos = pos.to_numpy()

        G.index = ct_list
        G.columns = ct_list
        Edges = nx.from_pandas_adjacency(G).edges()
        Nodes2pos = dict(zip(ct_list, list(pos)))

        # parameter setting
        fig = plt.figure(figsize=(width, height))
        ax = plt.subplot(1, 2, 1)
        # network
        color_list = plt.get_cmap(cmap, len(ct_list))
        ax.scatter(pos[:, 0], pos[:, 1], c=color_list.colors, zorder=1, s=dot_size)
        for i in range(len(ct_list)):
            ax.text(pos[i, 0], pos[i, 1], s=ct_list[i], zorder=2)
        for i, j in Edges:
            xi, yi = Nodes2pos[i]
            xj, yj = Nodes2pos[j]
            ax.plot([xi, xj], [yi, yj], color='black', zorder=0)

        # plotting
        ax = plt.subplot(1, 2, 2)

        cell_pos = df[[0, 1]].to_numpy()

        if color in self.stereo_exp_data.cells:
            if self.stereo_exp_data.cells[color].dtype == 'category':
                ax.scatter(cell_pos[:, 0], cell_pos[:, 1], s=size,
                           c=self.stereo_exp_data.cells[color].cat.codes.to_numpy(), cmap=cmap)
                cell_pos_df = pd.DataFrame(cell_pos)
                cell_pos_df[color] = self.stereo_exp_data.cells[color].to_list()
                cell_center_pos_df = cell_pos_df.groupby(color).mean()
                for s, row in cell_center_pos_df.iterrows():
                    ax.text(row[0], row[1], s)
            else:
                ax.scatter(cell_pos[:, 0], cell_pos[:, 1], s=size, c=self.stereo_exp_data.cells[color])
        elif color in self.stereo_exp_data.gene_names:
            gene_list = list(self.stereo_exp_data.genes.to_df().index)
            gene_index = gene_list.index(color)
            clist = self.stereo_exp_data.exp_matrix[:, gene_index]
            ax.scatter(cell_pos[:, 0], cell_pos[:, 1], s=size, c=clist)

        ax.set_xticks([])
        ax.set_yticks([])
        return fig
