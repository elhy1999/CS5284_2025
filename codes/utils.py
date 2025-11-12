# Load libraries
import numpy as np
import scipy.sparse # sparse matrix
import scipy.sparse.linalg
import warnings; warnings.filterwarnings("ignore")


def graph_laplacian(W: scipy.sparse.csr_matrix, normalization: str="sym"):
    """
    Compute the Laplacian matrix of the graph.
    Parameters
    ----------
    W : scipy.sparse.csr_matrix
        Adjacency matrix of the graph.
    normalization : str or None
        Type of normalization to use. Can be one of the following:
        - None: unnormalized Laplacian
        - "sym": symmetric normalized Laplacian
        - "rw": random-walk normalized Laplacian
    Returns
    -------
    L : scipy.sparse.csr_matrix
        Laplacian matrix of the graph.
    """
    
    # Degree vector
    d = W.sum(axis=0)

    # Laplacian matrix
    if not normalization:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    elif normalization == "sym":
        d += np.spacing(np.array(0, W.dtype)) # d += epsilon
        d = 1.0 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D
    elif normalization == "rw":
        d += np.spacing(np.array(0, W.dtype)) # d += epsilon
        d = 1.0 / d
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W
    return L