import sys
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from sklearn.covariance import empirical_covariance
from src.glasso_installation import check_and_install_glasso



# Activate the automatic conversion of numpy objects to R objects
numpy2ri.activate()

# Check and install glasso if needed before defining the R function
if not check_and_install_glasso():
    raise RuntimeError("Failed to ensure glasso package is available")

# Define the R function for weighted graphical lasso
ro.r('''
weighted_glasso <- function(data, penalty_matrix, nobs) {
    # Suppress warnings and messages when loading the library
    suppressWarnings(suppressMessages(library(glasso, quietly = TRUE)))
    
    tryCatch({
        result <- glasso(s = as.matrix(data), rho = penalty_matrix, nobs = nobs)
        return(list(precision_matrix = result$wi, edge_counts = result$wi != 0))
    }, error = function(e) {
        return(list(error_message = toString(e$message)))
    })
}
''')


def optimize_graph(data, prior_matrix, lambda_np, lambda_wp, verbose=False):
    """
    Optimizes the objective function using the entire data set and the estimated lambda.

    Parameters
    ----------
    data : array-like, shape (n, p)
        The data matrix.
    prior_matrix : array-like, shape (p, p)
        The prior matrix.
    lambda_val : float
        The regularization parameter for the edges.

    Returns
    -------
    opt_precision_mat : array-like, shape (p, p)
        The optimized precision matrix.
    """
    # Number of observations
    nobs = data.shape[0]
    p = data.shape[1]

    complete_graph_edges = (p * (p - 1)) / 2

    try:
        S = empirical_covariance(data)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr, flush=True)
        return np.zeros((p, p)), np.zeros((p, p)), 0


    # generate penalty matrix, where values = lambda_np for non-prior edges and lambda_wp for prior edges
    penalty_matrix = np.zeros_like(prior_matrix, dtype=np.float64)

    # prior_matrix = prior_matrix.astype(int)


    # Assign penalties based on the prior matrix
    penalty_matrix[prior_matrix != 0] = lambda_wp
    penalty_matrix[prior_matrix == 0] = lambda_np

    # # fill diagonal with 0s
    np.fill_diagonal(penalty_matrix, 0)

    # check for NaNs or Infs in penalty matrix
    if np.isnan(penalty_matrix).any():
        print('NaNs in penalty matrix')
    if np.isinf(penalty_matrix).any():
        print('Infs in penalty matrix')


    # print(f'P: {p}')
    # Call the R function from Python
    weighted_glasso = ro.globalenv['weighted_glasso']
    try:
        result = weighted_glasso(S, penalty_matrix, nobs)   
        # Check for an error message returned from R
        if 'error_message' in result.names:
            error_message = result.rx('error_message')[0][0]
            print(f"R Error: {error_message}", file=sys.stderr, flush=True)
            return np.zeros((p, p)), np.zeros((p, p)), 0
        else:
            precision_matrix = np.array(result.rx('precision_matrix')[0])
            np.fill_diagonal(precision_matrix, 0)
            edge_counts = np.sum((np.abs(precision_matrix) > 1e-5).astype(int)) / 2 # EDGE_DIVIDER
            density = edge_counts / complete_graph_edges
            return precision_matrix, edge_counts, density
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr, flush=True)
        return np.zeros((p, p)), np.zeros((p, p)), 0



def evaluate_reconstruction(adj_matrix, opt_precision_mat, threshold=1e-5):
    """
    Evaluate the accuracy of the reconstructed adjacency matrix.

    Parameters
    ----------
    adj_matrix : array-like, shape (p, p)
        The original adjacency matrix.
    opt_precision_mat : array-like, shape (p, p)
        The optimized precision matrix.
    threshold : float, optional
        The threshold for considering an edge in the precision matrix. Default is 1e-5.

    Returns
    -------
    metrics : dict
        Dictionary containing precision, recall, f1_score, and jaccard_similarity.
    """
    # Convert the optimized precision matrix to binary form
    reconstructed_adj = (np.abs(opt_precision_mat) > threshold).astype(int)
    np.fill_diagonal(reconstructed_adj, 0)
    np.fill_diagonal(adj_matrix, 0)

    # True positives, false positives, etc.
    tp = np.sum((reconstructed_adj == 1) & (adj_matrix == 1))
    fp = np.sum((reconstructed_adj == 1) & (adj_matrix == 0))
    fn = np.sum((reconstructed_adj == 0) & (adj_matrix == 1))
    tn = np.sum((reconstructed_adj == 0) & (adj_matrix == 0))

    # Precision, recall, F1 score
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # Jaccard similarity
    jaccard_similarity = tp / (tp + fp + fn)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'jaccard_similarity': jaccard_similarity
    }

    return metrics
