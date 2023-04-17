import numpy as np
import scipy as sp

from anndata import AnnData
from typing import Optional
# from scanpy.neighbors import Neighbors
from scipy.sparse.csgraph import minimum_spanning_tree

from stereo.algorithm.neighbors import find_neighbors

from  .. import _utils
from stereo.log_manager import logger

from .. _compat import Literal


_AVAIL_MODELS = {'v1.0', 'v1.2'}


def paga(
    adata: AnnData,
    groups: Optional[str] = None,
    use_rna_velocity: bool = False,
    model: Literal['v1.2', 'v1.0'] = 'v1.2',
    neighbors_key: Optional[str] = None,
    copy: bool = False,
):
    """\
    Mapping out the coarse-grained connectivity structures of complex manifolds [Wolf19]_.

    By quantifying the connectivity of partitions (groups, clusters) of the
    single-cell graph, partition-based graph abstraction (PAGA) generates a much
    simpler abstracted graph (*PAGA graph*) of partitions, in which edge weights
    represent confidence in the presence of connections. By tresholding this
    confidence in :func:`~scanpy.pl.paga`, a much simpler representation of the
    manifold data is obtained, which is nonetheless faithful to the topology of
    the manifold.

    The confidence should be interpreted as the ratio of the actual versus the
    expected value of connetions under the null model of randomly connecting
    partitions. We do not provide a p-value as this null model does not
    precisely capture what one would consider "connected" in real data, hence it
    strongly overestimates the expected value. See an extensive discussion of
    this in [Wolf19]_.

    .. note::
        Note that you can use the result of :func:`~scanpy.pl.paga` in
        :func:`~scanpy.tl.umap` and :func:`~scanpy.tl.draw_graph` via
        `init_pos='paga'` to get single-cell embeddings that are typically more
        faithful to the global topology.

    Parameters
    ----------
    adata
        An annotated data matrix.
    groups
        Key for categorical in `adata.obs`. You can pass your predefined groups
        by choosing any categorical annotation of observations. Default:
        The first present key of `'leiden'` or `'louvain'`.
    use_rna_velocity
        Use RNA velocity to orient edges in the abstracted graph and estimate
        transitions. Requires that `adata.uns` contains a directed single-cell
        graph with key `['velocity_graph']`. This feature might be subject
        to change in the future.
    model
        The PAGA connectivity model.
    neighbors_key
        If not specified, paga looks `.uns['neighbors']` for neighbors settings
        and `.obsp['connectivities']`, `.obsp['distances']` for connectivities and
        distances respectively (default storage places for `pp.neighbors`).
        If specified, paga looks `.uns[neighbors_key]` for neighbors settings and
        `.obsp[.uns[neighbors_key]['connectivities_key']]`,
        `.obsp[.uns[neighbors_key]['distances_key']]` for connectivities and distances
        respectively.
    copy
        Copy `adata` before computation and return a copy. Otherwise, perform
        computation inplace and return `None`.

    Returns
    -------
    **connectivities** : :class:`numpy.ndarray` (adata.uns['connectivities'])
        The full adjacency matrix of the abstracted graph, weights correspond to
        confidence in the connectivities of partitions.
    **connectivities_tree** : :class:`scipy.sparse.csr_matrix` (adata.uns['connectivities_tree'])
        The adjacency matrix of the tree-like subgraph that best explains
        the topology.

    Notes
    -----
    Together with a random walk-based distance measure
    (e.g. :func:`scanpy.tl.dpt`) this generates a partial coordinatization of
    data useful for exploring and explaining its variation.

    .. currentmodule:: scanpy

    See Also
    --------
    pl.paga
    pl.paga_path
    pl.paga_compare
    """
    check_neighbors = 'neighbors' if neighbors_key is None else neighbors_key
    if check_neighbors not in adata.uns:
        raise ValueError(
            'You need to run `pp.neighbors` first to compute a neighborhood graph.'
        )
    if groups is None:
        for k in ("leiden", "louvain"):
            if k in adata.obs.columns:
                groups = k
                break
    if groups is None:
        raise ValueError(
            'You need to run `tl.leiden` or `tl.louvain` to compute '
            "community labels, or specify `groups='an_existing_key'`"
        )
    elif groups not in adata.obs.columns:
        raise KeyError(f'`groups` key {groups!r} not found in `adata.obs`.')

    adata = adata.copy() if copy else adata
    _utils.sanitize_anndata(adata)
    start = logger.info('running PAGA')
    paga = PAGA(adata, groups, model=model, neighbors_key=neighbors_key)
    # only add if not present
    if 'paga' not in adata.uns:
        adata.uns['paga'] = {}
    if not use_rna_velocity:
        paga.compute_connectivities()
        adata.uns['paga']['connectivities'] = paga.connectivities
        adata.uns['paga']['connectivities_tree'] = paga.connectivities_tree
        # adata.uns['paga']['expected_n_edges_random'] = paga.expected_n_edges_random
        adata.uns[groups + '_sizes'] = np.array(paga.ns)
    else:
        paga.compute_transitions()
        adata.uns['paga']['transitions_confidence'] = paga.transitions_confidence
        # adata.uns['paga']['transitions_ttest'] = paga.transitions_ttest

    adata.uns['paga']['groups'] = groups
    # logger.info(
    #     '    finished',
    #     time=start,
    #     deep='added\n'
    #     + (
    #         "    'paga/transitions_confidence', connectivities adjacency (adata.uns)"
    #         # "    'paga/transitions_ttest', t-test on transitions (adata.uns)"
    #         if use_rna_velocity
    #         else "    'paga/connectivities', connectivities adjacency (adata.uns)\n"
    #         "    'paga/connectivities_tree', connectivities subtree (adata.uns)"
    #     ),
    # )

    if use_rna_velocity:
        logger.info('    finished\n'
                    + ' \'paga/transitions_confidence\', connectivities adjacency (adata.uns)\n'
                    + '\'paga/connectivities_tree\', connectivities subtree (adata.uns)')
    else:
        logger.info('    finished\n'
                    + ' \'paga/connectivities\', connectivities adjacency (adata.uns)\n'
                    + '\'paga/connectivities_tree\', connectivities subtree (adata.uns)')

    return adata if copy else None


class PAGA:
    def __init__(self, adata, groups, model='v1.2', neighbors_key=None):
        assert groups in adata.obs.columns
        self._adata = adata

        # do import here to avoid circular import
        from ..neighbors import Neighbors

        self._neighbors = Neighbors(adata, neighbors_key=neighbors_key)
        self._model = model
        self._groups_key = groups

    def compute_connectivities(self):
        if self._model == 'v1.2':
            return self._compute_connectivities_v1_2()
        elif self._model == 'v1.0':
            return self._compute_connectivities_v1_0()
        else:
            raise ValueError(
                f'`model` {self._model} needs to be one of {_AVAIL_MODELS}.'
            )

    def _compute_connectivities_v1_2(self):
        import igraph

        ones = self._neighbors.distances.copy()

        ones.data = np.ones(len(ones.data))
        # should be directed if we deal with distances
        g = _utils.get_igraph_from_adjacency(ones, directed=True)
        vc = igraph.VertexClustering(
            g, membership=self._adata.obs[self._groups_key].cat.codes.values
        )
        ns = vc.sizes()
        n = sum(ns)
        es_inner_cluster = [vc.subgraph(i).ecount() for i in range(len(ns))]
        cg = vc.cluster_graph(combine_edges='sum')

        inter_es = _utils.get_sparse_from_igraph(cg, weight_attr='weight')
        es = np.array(es_inner_cluster) + inter_es.sum(axis=1).A1
        inter_es = inter_es + inter_es.T  # \epsilon_i + \epsilon_j
        connectivities = inter_es.copy()
        expected_n_edges = inter_es.copy()
        inter_es = inter_es.tocoo()
        for i, j, v in zip(inter_es.row, inter_es.col, inter_es.data):
            expected_random_null = (es[i] * ns[j] + es[j] * ns[i]) / (n - 1)
            if expected_random_null != 0:
                scaled_value = v / expected_random_null
            else:
                scaled_value = 1
            if scaled_value > 1:
                scaled_value = 1
            connectivities[i, j] = scaled_value
            expected_n_edges[i, j] = expected_random_null
        # set attributes
        self.ns = ns
        self.expected_n_edges_random = expected_n_edges
        self.connectivities = connectivities
        self.connectivities_tree = self._get_connectivities_tree_v1_2()
        return inter_es.tocsr(), connectivities

    def _compute_connectivities_v1_0(self):
        import igraph

        ones = self._neighbors.connectivities.copy()

        ones.data = np.ones(len(ones.data))
        g = _utils.get_igraph_from_adjacency(ones)
        vc = igraph.VertexClustering(
            g, membership=self._adata.obs[self._groups_key].cat.codes.values
        )  # membership: 每个ele按顺序得到index
        ns = vc.sizes()
        cg = vc.cluster_graph(combine_edges='sum')
        inter_es = _utils.get_sparse_from_igraph(cg, weight_attr='weight') / 2
        connectivities = inter_es.copy()
        inter_es = inter_es.tocoo()
        n_neighbors_sq = self._neighbors.n_neighbors**2
        for i, j, v in zip(inter_es.row, inter_es.col, inter_es.data):
            # have n_neighbors**2 inside sqrt for backwards compat
            geom_mean_approx_knn = np.sqrt(n_neighbors_sq * ns[i] * ns[j])
            if geom_mean_approx_knn != 0:
                scaled_value = v / geom_mean_approx_knn
            else:
                scaled_value = 1
            connectivities[i, j] = scaled_value
        # set attributes
        self.ns = ns
        self.connectivities = connectivities
        self.connectivities_tree = self._get_connectivities_tree_v1_0(inter_es)
        return inter_es.tocsr(), connectivities  # index对应categories的标号

    def _get_connectivities_tree_v1_2(self):
        inverse_connectivities = self.connectivities.copy()
        inverse_connectivities.data = 1.0 / inverse_connectivities.data
        # todo: print
        connectivities_tree = minimum_spanning_tree(inverse_connectivities)
        connectivities_tree_indices = [
            connectivities_tree[i].nonzero()[1]
            for i in range(connectivities_tree.shape[0])
        ]
        connectivities_tree = sp.sparse.lil_matrix(
            self.connectivities.shape, dtype=float
        )
        for i, neighbors in enumerate(connectivities_tree_indices):
            if len(neighbors) > 0:
                connectivities_tree[i, neighbors] = self.connectivities[i, neighbors]
        return connectivities_tree.tocsr()

    def _get_connectivities_tree_v1_0(self, inter_es):
        inverse_inter_es = inter_es.copy()
        inverse_inter_es.data = 1.0 / inverse_inter_es.data
        connectivities_tree = minimum_spanning_tree(inverse_inter_es)
        connectivities_tree_indices = [
            connectivities_tree[i].nonzero()[1]
            for i in range(connectivities_tree.shape[0])
        ]
        connectivities_tree = sp.sparse.lil_matrix(inter_es.shape, dtype=float)
        for i, neighbors in enumerate(connectivities_tree_indices):
            if len(neighbors) > 0:
                connectivities_tree[i, neighbors] = self.connectivities[i, neighbors]
        return connectivities_tree.tocsr()

    def compute_transitions(self):
        vkey = 'velocity_graph'
        if vkey not in self._adata.uns:
            if 'velocyto_transitions' in self._adata.uns:
                self._adata.uns[vkey] = self._adata.uns['velocyto_transitions']
                logger.debug(
                    "The key 'velocyto_transitions' has been changed to 'velocity_graph'."
                )
            else:
                raise ValueError(
                    'The passed AnnData needs to have an `uns` annotation '
                    "with key 'velocity_graph' - a sparse matrix from RNA velocity."
                )
        if self._adata.uns[vkey].shape != (self._adata.n_obs, self._adata.n_obs):
            raise ValueError(
                f"The passed 'velocity_graph' have shape {self._adata.uns[vkey].shape} "
                f"but shoud have shape {(self._adata.n_obs, self._adata.n_obs)}"
            )
        # restore this at some point
        # if 'expected_n_edges_random' not in self._adata.uns['paga']:
        #     raise ValueError(
        #         'Before running PAGA with `use_rna_velocity=True`, run it with `False`.')
        import igraph

        g = _utils.get_igraph_from_adjacency(
            self._adata.uns[vkey].astype('bool'),
            directed=True,
        )
        vc = igraph.VertexClustering(
            g, membership=self._adata.obs[self._groups_key].cat.codes.values
        )
        # set combine_edges to False if you want self loops
        cg_full = vc.cluster_graph(combine_edges='sum')
        transitions = _utils.get_sparse_from_igraph(cg_full, weight_attr='weight')
        transitions = transitions - transitions.T
        transitions_conf = transitions.copy()
        transitions = transitions.tocoo()
        total_n = self._neighbors.n_neighbors * np.array(vc.sizes())
        # total_n = self._n_neighbors * np.array(vc.sizes())
        # total_n_sum = sum(total_n)
        # expected_n_edges_random = self._adata.uns['paga']['expected_n_edges_random']
        for i, j, v in zip(transitions.row, transitions.col, transitions.data):
            # if expected_n_edges_random[i, j] != 0:
            #     # factor 0.5 because of asymmetry
            #     reference = 0.5 * expected_n_edges_random[i, j]
            # else:
            #     # approximate
            #     reference = self._neighbors.n_neighbors * total_n[i] * total_n[j] / total_n_sum
            reference = np.sqrt(total_n[i] * total_n[j])
            transitions_conf[i, j] = 0 if v < 0 else v / reference
        transitions_conf.eliminate_zeros()
        # transpose in order to match convention of stochastic matrices
        # entry ij means transition from j to i
        self.transitions_confidence = transitions_conf.T