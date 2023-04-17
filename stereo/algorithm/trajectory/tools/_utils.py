from .. import settings
from stereo.log_manager import logger
from ._pca import pca

doc_use_rep = """\
use_rep
    Use the indicated representation. `'X'` or any key for `.obsm` is valid.
    If `None`, the representation is chosen automatically:
    For `.n_vars` < 50, `.X` is used, otherwise 'X_pca' is used.
    If 'X_pca' is not present, it’s computed with default parameters.\
"""

doc_n_pcs = """\
n_pcs
    Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`.\
"""


def _choose_representation(adata, use_rep=None, n_pcs=None, silent=False):
    verbosity = settings.verbosity
    if silent and settings.verbosity > 1:
        settings.verbosity = 1
    if use_rep is None and n_pcs == 0:  # backwards compat for specifying `.X`
        use_rep = 'X'
    if use_rep is None:
        if adata.n_vars > settings.N_PCS:
            if 'X_pca' in adata.obsm.keys():
                if n_pcs is not None and n_pcs > adata.obsm['X_pca'].shape[1]:
                    raise ValueError(
                        '`X_pca` does not have enough PCs. Rerun `sc.pp.pca` with adjusted `n_comps`.'
                    )
                X = adata.obsm['X_pca'][:, :n_pcs]
                logger.info(f'    using \'X_pca\' with n_pcs = {X.shape[1]}')
            else:
                logger.warning(
                    f'You’re trying to run this on {adata.n_vars} dimensions of `.X`, '
                    'if you really want this, set `use_rep=\'X\'`.\n         '
                    'Falling back to preprocessing with `sc.pp.pca` and default params.'
                )
                X = pca(adata.X)
                adata.obsm['X_pca'] = X[:, :n_pcs]
        else:
            logger.info('    using data matrix X directly')
            X = adata.X
    else:
        if use_rep in adata.obsm.keys() and n_pcs is not None:
            if n_pcs > adata.obsm[use_rep].shape[1]:
                raise ValueError(
                    f'{use_rep} does not have enough Dimensions. Provide a '
                    'Representation with equal or more dimensions than'
                    '`n_pcs` or lower `n_pcs` '
                )
            X = adata.obsm[use_rep][:, :n_pcs]
        elif use_rep in adata.obsm.keys() and n_pcs is None:
            X = adata.obsm[use_rep]
        elif use_rep == 'X':
            X = adata.X
        else:
            raise ValueError(
                'Did not find {} in `.obsm.keys()`. '
                'You need to compute it first.'.format(use_rep)
            )
    settings.verbosity = verbosity  # resetting verbosity
    return X
