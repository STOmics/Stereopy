from typing import Optional

import numpy as np
import scipy as sp
from scipy.sparse.csgraph import minimum_spanning_tree

from stereo.log_manager import logger
from .algorithm_base import AlgorithmBase

_AVAIL_MODELS = {'v1.0', 'v1.2'}


def _get_igraph_from_adjacency(adjacency, directed=None):
    """
    Get igraph graph from adjacency matrix.
    """
    import igraph as ig

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1

    g = ig.Graph(directed=directed)
    # this adds adjacency.shape[0] vertices
    g.add_vertices(adjacency.shape[0])
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


def _get_sparse_from_igraph(graph, weight_attr=None):
    from scipy.sparse import csr_matrix

    edges = graph.get_edgelist()
    if weight_attr is None:
        weights = [1] * len(edges)
    else:
        weights = graph.es[weight_attr]
    if not graph.is_directed():
        edges.extend([(v, u) for u, v in edges])
        weights.extend(weights)
    shape = graph.vcount()
    shape = (shape, shape)
    if len(edges) > 0:
        return csr_matrix((weights, zip(*edges)), shape=shape)
    else:
        return csr_matrix(shape)


class Paga(AlgorithmBase):

    def main(self, groups: Optional[str] = None, use_rna_velocity: bool = False, model='v1.2',
             neighbors_key: Optional[str] = "neighbors", copy: bool = False, **kwargs):
        """
        Partition-based graph abstraction, shorted as PAGA, is an algorithm which provides an interpretable graph-like
        map of the arising data manifold, based on estimating connectivity of manifold partitions.

        Parameters
        ----------
        groups
            Key for categorical in `stereo_exp_data.obs`. You can pass your predefined groups
            by choosing any categorical annotation of observations. Default:
            The first present key of `'leiden'` or `'louvain'`.
        use_rna_velocity
            TODO : not finished yet
        model
            Default to `v1.2`. The `PAGA` connectivity model.
        neighbors_key
            If not specified, paga looks `.uns['neighbors']` for neighbors settings
            and `.obsp['connectivities']`, `.obsp['distances']` for connectivities and
            distances respectively (default storage places for `pp.neighbors`).
            If specified, paga looks `.uns[neighbors_key]` for neighbors settings and
            `.obsp[.uns[neighbors_key]['connectivities_key']]`,
            `.obsp[.uns[neighbors_key]['distances_key']]` for connectivities and distances
            respectively.
        copy
            Copy `stereo_exp_data` before computation and return a copy. Otherwise, perform
            computation inplace and return `None`.

        Returns
        -------
            return `stereo_exp_data` if `copy` is `True`, or `None` if `copy` is `False`.
        """

        if groups is None:
            # TODO using a common define cluster method enum set
            for k in ("leiden", "louvain", "phenograph", "annotation"):
                if k in self.stereo_exp_data.cells:
                    groups = k
                    logger.info(f"`groups` is None, automatically choose {groups} as `groups`")
                    break
        if groups is None:
            raise ValueError("you need to run cluster method to compute community labels, or specify `groups`")
        elif groups not in self.stereo_exp_data.cells:
            raise KeyError(f'`groups` key {groups} not found in `stereo_exp_data.cells`')

        neighbors_key = 'neighbors' if neighbors_key is None else neighbors_key
        # TODO still using old way to check `neighbor`
        if neighbors_key not in self.stereo_exp_data.tl.result:
            raise ValueError('you need to run `tl.neighbors` first to compute a neighborhood graph')

        # TODO this might be a base level method
        if copy:
            from copy import deepcopy
            stereo_exp_data = deepcopy(self.stereo_exp_data)
        else:
            stereo_exp_data = self.stereo_exp_data

        # TODO old code from pr, need check
        # _utils.sanitize_anndata(stereo_exp_data)

        paga_alg_obj = PAGA(stereo_exp_data, groups, model=model, neighbors_key=neighbors_key)
        # only add if not present
        paga_res_dict = {}
        if not use_rna_velocity:
            paga_alg_obj.compute_connectivities()
            paga_res_dict['connectivities'] = paga_alg_obj.connectivities
            paga_res_dict['connectivities_tree'] = paga_alg_obj.connectivities_tree
            paga_res_dict[groups + '_sizes'] = np.array(paga_alg_obj.ns)
        else:
            paga_alg_obj.compute_transitions()
            paga_res_dict['transitions_confidence'] = paga_alg_obj.transitions_confidence

        paga_res_dict['groups'] = groups
        stereo_exp_data.tl.result['paga'] = paga_res_dict

        return stereo_exp_data if copy else None


class PAGA:
    def __init__(self, stereo_exp_data, groups, model='v1.2', neighbors_key=None):
        assert groups in stereo_exp_data.cells
        self._stereo_exp_data = stereo_exp_data

        # do import here to avoid circular import

        self._neighbors = {
            "neighbor": {"n_neighbors": 15},
            "connectivities": stereo_exp_data.tl.result['neighbors']['connectivities'],
            "distances": stereo_exp_data.tl.result['neighbors']['nn_dist']
        }
        self._model = model
        self._groups_key = groups
        self._stereo_exp_data.cells[self._groups_key] = self._stereo_exp_data.cells[self._groups_key].astype('category')

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

        ones = self._neighbors['distances'].copy()
        ones.data = np.ones(len(ones.data))
        # should be directed if we deal with distances
        g = _get_igraph_from_adjacency(ones, directed=True)
        vc = igraph.VertexClustering(
            g, membership=self._stereo_exp_data.cells[self._groups_key].cat.codes.values
        )
        ns = vc.sizes()
        n = sum(ns)
        es_inner_cluster = [vc.subgraph(i).ecount() for i in range(len(ns))]
        cg = vc.cluster_graph(combine_edges='sum')

        inter_es = _get_sparse_from_igraph(cg, weight_attr='weight')
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

        ones = self._neighbors['connectivities'].copy()

        ones.data = np.ones(len(ones.data))
        g = _get_igraph_from_adjacency(ones)
        vc = igraph.VertexClustering(
            g, membership=self._stereo_exp_data.cells[self._groups_key].cat.codes.values
        )  # membership: 每个ele按顺序得到index
        ns = vc.sizes()
        cg = vc.cluster_graph(combine_edges='sum')

        inter_es = _get_sparse_from_igraph(cg, weight_attr='weight') / 2
        connectivities = inter_es.copy()
        inter_es = inter_es.tocoo()
        n_neighbors_sq = self._neighbors["neighbor"]["n_neighbors"] ** 2
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
        if vkey not in self._stereo_exp_data.cells_pairwise:
            if 'velocyto_transitions' in self._stereo_exp_data.cells_pairwise:
                self._stereo_exp_data.cells_pairwise[vkey] = self._stereo_exp_data.cells_pairwise[
                    'velocyto_transitions']
                logger.debug(
                    "The key 'velocyto_transitions' has been changed to 'velocity_graph'."
                )
            else:
                raise ValueError(
                    'The passed AnnData needs to have an `uns` annotation '
                    "with key 'velocity_graph' - a sparse matrix from RNA velocity."
                )
        if self._stereo_exp_data.cells_pairwise[vkey].shape != (
                self._stereo_exp_data.cell_names, self._stereo_exp_data.cell_names):
            raise ValueError(
                f"The passed 'velocity_graph' have shape {self._stereo_exp_data.cells_pairwise[vkey].shape} "
                f"but shoud have shape {(self._stereo_exp_data.cell_names, self._stereo_exp_data.cell_names)}"
            )
        import igraph

        g = _get_igraph_from_adjacency(
            self._stereo_exp_data.cells_pairwise[vkey].astype('bool'),
            directed=True,
        )
        vc = igraph.VertexClustering(
            g, membership=self._stereo_exp_data.cells[self._groups_key].cat.codes.values
        )
        # set combine_edges to False if you want self loops
        cg_full = vc.cluster_graph(combine_edges='sum')
        transitions = _get_sparse_from_igraph(cg_full, weight_attr='weight')
        transitions = transitions - transitions.T
        transitions_conf = transitions.copy()
        transitions = transitions.tocoo()
        total_n = self._neighbors["neighbor"]["n_neighbors"] * np.array(vc.sizes())
        for i, j, v in zip(transitions.row, transitions.col, transitions.data):
            reference = np.sqrt(total_n[i] * total_n[j])
            transitions_conf[i, j] = 0 if v < 0 else v / reference
        transitions_conf.eliminate_zeros()
        # transpose in order to match convention of stochastic matrices
        # entry ij means transition from j to i
        self.transitions_confidence = transitions_conf.T
