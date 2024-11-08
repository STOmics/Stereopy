from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy as np


def nearest_neighbors(coord, coords, n_neighbors=5):
    neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(coords)
    _, neighs = neigh.kneighbors(np.atleast_2d(coord))
    return neighs


def kmeans_centers(coords: np.ndarray, n_clusters: int = 2) -> np.ndarray:
    """
    Use kmeans to find cluster centers based on coordinates.

    Parameters
    ----------
    coords
        Coordinates stored in a 2D array.
    n_clusters
        The number of cluster centers.
        (Default: 2)

    Returns
    -------
    np.ndarray
        Centers of clusters.
    """
    cell_coordinates = coords
    kmeans = KMeans(n_clusters, random_state=10086)
    kmeans.fit(cell_coordinates)
    cluster_centers = kmeans.cluster_centers_
    print("kmeans cluster centers:")
    list(map(print, cluster_centers))
    return cluster_centers
