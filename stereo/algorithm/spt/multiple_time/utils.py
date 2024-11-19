import numpy as np
from scipy.stats import wasserstein_distance
from scipy.sparse import issparse
import ot
from anndata import AnnData

from stereo.core.stereo_exp_data import StereoExpData

from ..utils import get_cell_coordinates

#### Calculate spatial coordinates distances
def spatial_dist(data:StereoExpData, spatial_key:str='spatial',spa_method:str='euclidean'):
    """Calculate spatial distance.
    Args:
        adata:AnnData object.
        spatial_key: adata.obsm[spatial_key],corresponds to spatial coordinates.
        metric
    Rerurns:
        spatial distances. The dimension is n_obs*n_obs.
    """
    spa_coords = get_cell_coordinates(data, basis=spatial_key)
    spa_dist = ot.dist(spa_coords, spa_coords, metric=spa_method)
    return np.array(spa_dist) 


#### Get gene expression matrix from adata
def get_exp_matrix(adata:AnnData, layer:str ='X'):
    """Get gene expression matrix(dense not sparse) from adata object.
    Args:
        adata:AnnData object.
        layer: If ``'X'``, uses ``.X``, otherwise uses the representation given by ``adata.layers[layer]``.
    Rerurns:
        gene expression matrix. The dimension is n_obs*n_genes.
    """
    if layer == 'X':
        exp_matrix = adata.X
    else:
        exp_matrix = adata.layers[layer]
    if issparse(exp_matrix):
        exp_matrix = exp_matrix.toarray()
    else:
        exp_matrix = np.array(exp_matrix)
    return exp_matrix


#### Calculate gene expression dissimilarity

def wasserstein_distance(
    X:np.ndarray,
    Y:np.ndarray,
):
    """Compute Wasserstein distance between two gene expression matrix.
    
    Args:
        X: np.array with dim (n_obs * n_genes).
        Y: np.array with dim (m_obs * n_genes).
    Rerurns:
        W_D: np.array with dim(n_obs * m_obs). Wasserstein distance matrix.
        
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    W_D = np.zeros((X.shape[0],Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            dist = wasserstein_distance(X[i], Y[j])
            dist_matrix[i][j] = dist
    return W_D

def kl_divergence_backend(
    X:np.ndarray,
    Y:np.ndarray,
):
    """Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.
    Takes advantage of POT backend to speed up computation.
    
    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)
    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    nx = ot.backend.get_backend(X, Y)

    X = X / nx.sum(X, axis=1, keepdims=True)
    Y = Y / nx.sum(Y, axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum("ij,ij->i", X, log_X)
    X_log_X = nx.reshape(X_log_X, (1, X_log_X.shape[0]))
    KL_D = X_log_X.T - nx.dot(X, log_Y.T)
    return KL_D

def mcc_distance(
    X:np.ndarray,
    Y:np.ndarray,
):
    
    """Compute matthew's correlation coefficient between two gene expression matrix.
    Args:
        X: np.array with dim (n_obs * n_genes).
        Y: np.array with dim (m_obs * n_genes).
    
    """
    def __mcc(true_labels, pred_labels):
        TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
        TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
        FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
        FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
        mcc = (TP * TN) - (FP * FN)
        denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        if denom==0:
            return 0
        return mcc / denom
    
    cost = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            cost[i, j] = __mcc(X[i], Y[j])
    return cost



def gene_dist(
    X_1:np.ndarray,
    X_2:np.ndarray,
    gene_method:str='kl',
):
    """Calculate gene expression dissimilarity.
    
    Args:
        X: Gene expression matrix of adata_1. np.array with dim (n_obs * n_genes).
        Y: Gene expression matrix of adata_2. np.array with dim (n_obs * n_genes).
        method: calculate gene expression dissimilarity measure: ``'euclidean'`` or ``'cosine'``or``'mcc'``or``'wasserstein'``or``'kl'``.
    Rerurns:
        W_D: np.array with dim(n_obs * m_obs). Wasserstein distance matrix.
        
    """
    if gene_method == 'euclidean' or gene_method == 'cosine':
        dist_matrix = ot.dist(X_1,X_2,metric=gene_method)
    elif gene_method == 'mcc':
        dist_matrix = mcc_distance(X_1,X_2)

    elif gene_method == 'wasserstein':
        dist_matrix = wasserstein_distance(X_1,X_2) 
                
    elif gene_method == 'kl':
        s_X_1 = X_1 + 0.01
        s_X_2 = X_2 + 0.01
        dist_matrix = (kl_divergence_backend(s_X_1, s_X_2) + kl_divergence_backend(s_X_2, s_X_1).T) / 2

    return dist_matrix

##check adata type
def pre_check_adata(
    data: StereoExpData,
    spatial_key: str = 'spatial',
    time: str = 'time',
    annotation: str = 'celltype'
):
    """
    Check adata type. 
    """
    data.cells['time'] = data.cells[time]
    data.cells['annotation'] = data.cells[annotation]
    data.cells['x'] = data.cells_matrix[spatial_key][:,0]
    data.cells['y'] = data.cells_matrix[spatial_key][:,1]
    data.cells['cell_id'] = data.cell_names
    return data

## unbalanced ot and FGW.
## ualanced ot
## min  <gamma,c> + epsilon*H(gamma) 
##      + rho*KL(gamma|mu) + rho*KL(gamma^T|nu)
## s.t. gamma >= 0
def uot(mu, nu, c, epsilon,
         niter=50, tau=-0.5, verb = 1, rho = np.Inf, stopThr= 1E-7):

    lmbda = rho / ( rho + epsilon )
    if np.isinf(rho): lmbda = 1
    ## format mu(m,1) nu(n,1)
    mu = np.asarray(mu, float).reshape(-1,1)
    nu = np.asarray(nu, float).reshape(-1,1)
    N = [mu.shape[0], nu.shape[0]]
    H1 = np.ones([N[0],1])
    H2 = np.ones([N[1],1])
    ## initialize
    errs = []; Wprimal = []; Wdual = []
    u = np.zeros([N[0],1], float)
    v = np.zeros([N[1],1], float)
    for i in range(niter):
        u_prev = u
        ## update u,v,pi
        u = ave(tau, u, \
            lmbda * epsilon * np.log(mu) \
            - lmbda * epsilon * lse( M(u,v,H1,H2,c,epsilon) ) \
            + lmbda * u )
        v = ave(tau, v, \
            lmbda * epsilon * np.log(nu) \
            - lmbda * epsilon * lse( M(u,v,H1,H2,c,epsilon).T ) \
            + lmbda * v )
        pi = np.exp( M(u,v,H1,H2,c,epsilon) )
        ## evaluate the primal dual functions and errors
        if np.isinf(rho):
            Wprimal.append(np.sum(c * pi) - epsilon*H(pi) )
            Wdual.append(np.sum(u*mu) + np.sum(v*nu) - epsilon*np.sum(pi) )
            err = np.linalg.norm( np.sum(pi,axis=1) - mu )
            errs.append( err )
        else:
            Wprimal.append(np.sum(c*pi) - epsilon*H(pi) \
                           + rho*KL(np.sum(pi,axis=1), mu) \
                           + rho*KL(np.sum(pi,axis=0), nu) )
            Wdual.append(- rho*KLd(u/rho,mu) - rho*KLd(v/rho,nu) \
                         - epsilon*np.sum(pi) )
            err = np.linalg.norm(u-u_prev,1)
            errs.append( err )
        ## check convergence condition
        if err < stopThr and i > niter:
            break

    return pi


## FGW algorithm
# min  (1-alpha)*<pi,c> + alpha*<pi,c1,c2> 
#      + epsilon*H(pi) 
#      + rho*KL(pi|mu) + rho*KL(pi^T|nu)
# s.t. pi >= 0
def usot(mu, nu, c, c1, c2, alpha, epsilon = 0.1,
         niter = 10, gw_loss = 'square', rho = np.Inf):
    ## format mu(m,1) nu(n,1)
    mu = np.asarray(mu, float).reshape(-1,1)
    nu = np.asarray(nu, float).reshape(-1,1)
    ## initialize pi,cost
    pi0 = np.outer(mu, nu)
    pi_old = np.array(pi0, float)
    G = np.empty(c.shape, float)
    for i in range(niter):
        ## Construct loss
        G_w = ( 1.0 - alpha ) * c
        if gw_loss == 'square':
            fc1 = 0.5*c1**2; fc2 = 0.5*c2**2
            hc1 = c1; hc2 = c2
        constC1 = np.dot(np.dot(fc1, mu), np.ones(len(nu), float).reshape(1,-1))
        constC2 = np.dot(np.ones(len(mu)).reshape(-1,1), np.dot(nu.reshape(1,-1),fc2.T))
        constC = constC1 + constC2
        G_gw = alpha * 2.0 * (constC - np.dot(hc1, pi_old).dot(hc2.T))
        G[:,:] = G_w[:,:] + G_gw[:,:]
        ## ot or uot
        if np.isinf(rho):
            pi_tuta = ot.sinkhorn(mu.reshape(-1), nu.reshape(-1), G, epsilon)
        else:
            pi_tuta = uot(mu, nu, G, epsilon, rho = rho)
        # Line search for update
        CxC_tuta_minus_old = c1.dot(np.dot(pi_tuta-pi_old, c2))
        CxC_old = c1.dot(np.dot(pi_old, c2))
        a = -alpha*np.sum( CxC_tuta_minus_old * ( pi_tuta-pi_old ) )
        b = np.sum( ( (1.0-alpha)*c+alpha*constC-2.0*alpha*CxC_old ) * ( pi_tuta-pi_old) )
        if a > 0:
            tau_update = min(1.0,max(0.0,-0.5*b/a))
        elif a + b < 0:
            tau_update = 1.0
        else:
            tau_update = 0.0
        pi_new = (1.0-tau_update) * pi_old + tau_update * pi_tuta
        pi_old = pi_new
    return pi_new


def ave(tau, u, u_prev):
    return tau * u + ( 1 - tau ) * u_prev

def lse(A):
    return np.log(np.sum(np.exp(A),axis=1)).reshape(-1,1)

def H(p):
    return -np.sum( p * np.log(p+1E-20)-1 )

def KL(h,p):
    return np.sum( h * np.log( h/p ) - h + p )

def KLd(u,p):
    return np.sum( p * ( np.exp(-u) - 1 ) )

def M(u,v,H1,H2,c,epsilon):
    y = -c + np.matmul(u.reshape(-1,1), H2.reshape(1,-1)) + \
        np.matmul(H1.reshape(-1,1), v.reshape(1,-1))
    return y/epsilon