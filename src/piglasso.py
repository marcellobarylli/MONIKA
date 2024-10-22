import os
import sys
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (MONIKA)
project_dir = os.path.dirname(script_dir)
# Add the project directory to the Python path
sys.path.append(project_dir)
# Change the working directory to the project directory
os.chdir(project_dir)

import numpy as np
import pandas as pd
import networkx as nx
import math
import matplotlib.pyplot as plt
from random import sample
import random
from numpy.random import multivariate_normal
from scipy.special import comb, erf
import scipy.stats as stats
from scipy.linalg import block_diag, eigh, inv
from sklearn.covariance import empirical_covariance
import os
import rpy2.situation
os.environ["R_HOME"] = rpy2.situation.get_r_home()

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from mpi4py import MPI
from tqdm import tqdm
import pickle
import warnings
import argparse
from scipy.stats import skewnorm

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

class QJSweeper:
    """
    Class for parallel optimisation of the piGGM objective function, across Q sub-samples and J lambdas.

    Attributes
    ----------
    data : array-like, shape (n, p)
        The data matrix.
    prior_matrix : array-like, shape (p, p)
        The prior matrix. Used to identify which edges are penalized by lambda_wp.
    p : int
        The number of variables.

    Methods
    -------
    objective(precision_vector, S, lambda_np, lambda_wp, prior_matrix)
        The objective function for the piGGM optimization problem.

    optimize_for_q_and_j(params)
        Optimizes the objective function for a given sub-sample (q) and lambda (j).
        
    subsample_optimiser(b, Q, lambda_range)
        Optimizes the objective function for all sub-samples and lambda values, using optimize_for_q_and_j.
    """
    def __init__(self, data, prior_matrix, b, Q, rank=1, size=1, seed=42):
        self.data = data
        self.prior_matrix = prior_matrix
        self.p = data.shape[1]
        self.n = data.shape[0]
        self.Q = Q
        self.subsample_indices = self.get_subsamples_indices(self.n, b, Q, rank, size, seed=42)

    @staticmethod
    def generate_synth_data(p, n, fp_fn_chance, skew, synth_density=0.03, seed=42):
        """
        Generates a scale-free synthetic nework with desired synth_density, and then generates synthetic data based on the network.
        """
        random.seed(seed)
        np.random.seed(seed)

        density_params = {
            0.02: [(100, 1), (150, 2), (300, 3), (500, 5), (750, 8), (1000, 10)],
            0.03: [(100, 2), (150, 2), (300, 5), (500, 8), (750, 11), (1000, 15)],
            0.04: [(100, 2), (150, 3), (300, 6), (500, 10), (750, 15), (1000, 20)],
            0.05: [(100, 3), (150, 4), (300, 8), (500, 13), (750, 19), (1000, 25)],
            0.1: [(100, 5), (150, 8), (300, 15), (500, 25), (750, 38), (1000, 50)],
            0.15: [(100, 8), (150, 11), (300, 23), (500, 38), (750, 56), (1000, 75)],
            0.2: [(100, 10), (150, 15), (300, 30), (500, 50), (750, 75), (1000, 100)]
        }


        # Determine m based on p and the desired synth_density
        m = 20  # Default value if p > 1000
        closest_distance = float('inf')
        for size_limit, m_value in density_params[synth_density]:
            distance = abs(p - size_limit)
            if distance < closest_distance:
                closest_distance = distance
                m = m_value
        
        
        # TRUE NETWORK
        G = nx.barabasi_albert_graph(p, m, seed=seed)
        adj_matrix = nx.to_numpy_array(G)
        np.fill_diagonal(adj_matrix, 0)


        # check if adj matrix is symmetric
        if not np.allclose(adj_matrix, adj_matrix.T, atol=1e-8):
            print('Adjacency matrix is not symmetric')
            sys.exit(1)

        
        # PRECISION MATRIX
        precision_matrix = np.copy(adj_matrix)

        # Try adding a small constant to the diagonal until the matrix is positive definite
        small_constant = 0.01
        is_positive_definite = False
        while not is_positive_definite:
            np.fill_diagonal(precision_matrix, precision_matrix.diagonal() + small_constant)
            eigenvalues = np.linalg.eigh(precision_matrix)[0]
            is_positive_definite = np.all(eigenvalues > 0)
            small_constant += 0.01  # Increment the constant

        # Compute the scaling factors for each variable (square root of the diagonal of the precision matrix)
        scaling_factors = np.sqrt(np.diag(precision_matrix))
        # Scale the precision matrix
        adjusted_precision = np.outer(1 / scaling_factors, 1 / scaling_factors) * precision_matrix

        covariance_mat = inv(adjusted_precision)

        # PRIOR MATRIX

        # Count the total number of edges in adj_matrix
        total_edges = np.sum(adj_matrix) // 2  # divide by 2 for undirected graph
        # print('Total edges: ', total_edges)

        # Calculate the number of edges to flip (25% of total edges)
        num_edges_to_flip = int(total_edges * fp_fn_chance)

        # Create a copy of adj_matrix to start forming prior_matrix
        prior_matrix = np.copy(adj_matrix)

        # Initialize a set to keep track of modified edges
        modified_edges = set()

        # Randomly select edges to turn into FNs (1 to 0)
        edges = np.transpose(np.where(adj_matrix == 1))
        np.random.shuffle(edges)

        # print('Number of edges: ', len(edges))

        num_edges_to_flip = min(num_edges_to_flip, len(edges))
        # print('NUMETOFLIP: ', num_edges_to_flip)

        flipped_fns = 0
        i = 0
        while flipped_fns < num_edges_to_flip and i < len(edges):
            x, y = edges[i]
            edge_tuple = (min(x, y), max(x, y))

            if edge_tuple not in modified_edges:
                prior_matrix[x, y] = 0
                prior_matrix[y, x] = 0
                modified_edges.add(edge_tuple)
                flipped_fns += 1

            i += 1

        # print('FNs: ', flipped_fns, ' / ', total_edges, ' edges flipped to FNs')


        # Randomly select non-edges to turn into FPs (0 to 1)
        non_edges = np.transpose(np.where(adj_matrix == 0))
        np.random.shuffle(non_edges)

        # Variable to keep track of the number of FPs added
        
        flipped_fps = 0
        i = 0
        while flipped_fps < num_edges_to_flip and i < len(non_edges):
            x, y = non_edges[i]
            edge_tuple = (min(x, y), max(x, y))

            if x != y and edge_tuple not in modified_edges:
                prior_matrix[x, y] = 1
                prior_matrix[y, x] = 1
                modified_edges.add(edge_tuple)
                # fps.add(edge_tuple)
                flipped_fps += 1

            i += 1


        # print('FPs: ', flipped_fps, ' / ', total_edges, ' non-edges flipped to FPs')


        # Ensure diagonal remains 0
        np.fill_diagonal(prior_matrix, 0)
        # np.fill_diagonal(adj_matrix, 0)


        # DATA MATRIX
        data = multivariate_normal(mean=np.zeros(G.number_of_nodes()), cov=covariance_mat, size=n)

        if skew != 0:
            print('APPLYING SKEW: ', skew)
            # Determining which columns to skew
            columns_to_skew = np.random.choice(data.shape[1], size=int(0.2 * data.shape[1]), replace=False)
            left_skew_columns = columns_to_skew[:len(columns_to_skew) // 2]
            right_skew_columns = columns_to_skew[len(columns_to_skew) // 2:]

            # Applying skewness
            for col in left_skew_columns:
                data[:, col] += skewnorm.rvs(-skew, size=n)  # Left skew
            for col in right_skew_columns:
                data[:, col] += skewnorm.rvs(skew, size=n)  # Right skew

            # add outliers
            num_outliers = int(0.05 * data.shape[0])
            outlier_indices = np.random.choice(data.shape[0], size=num_outliers, replace=False)
            outlier_columns = np.random.choice(data.shape[1], size=num_outliers, replace=True)
            data[outlier_indices, outlier_columns] += np.random.normal(loc=0, scale=2, size=num_outliers)


        return data, prior_matrix, adj_matrix
    
    def get_subsamples_indices(self, n, b, Q, rank, size, seed=42):
        """
        Generate a unique set of subsamples indices for a given MPI rank and size.
        """
        # Error handling: check if b and Q are valid 
        if b >= n:
            raise ValueError("b should be less than the number of samples n.")
        if Q > comb(n, b, exact=True):
            raise ValueError("Q should be smaller or equal to the number of possible sub-samples.")

        random.seed(seed + rank)  # Ensure each rank gets different subsamples
        subsamples_indices = set()

        # Each rank will attempt to generate Q/size unique subsamples
        subsamples_per_rank = Q // size
        attempts = 0
        max_attempts = 10e+5  # to avoid an infinite loop

        while len(subsamples_indices) < subsamples_per_rank and attempts < max_attempts:
            # Generate a random combination
            new_comb = tuple(sorted(sample(range(n), b)))
            subsamples_indices.add(new_comb)
            attempts += 1

        if attempts == max_attempts:
            raise Exception(f"Rank {rank}: Max attempts reached when generating subsamples.")

        return list(subsamples_indices)

    def optimize_for_q_and_j(self, single_subsamp_idx, lambdax):
        """
        Optimizes the objective function for a given sub-sample (q) and lambda (j).
        Parameters
        ----------
        subsamples_indices : array-like, shape (b)
            The indices of the sub-sample.
        lambdax : float
            The lambda value.

        Returns
        -------
        selected_sub_idx : array-like, shape (b)
            The indices of the sub-sample.
        lambdax : float
            The lambda value.
        edge_counts : array-like, shape (p, p)
            The edge counts of the optimized precision matrix.
        """
        data = self.data
        p = self.p
        prior_matrix = self.prior_matrix
        sub_sample = data[np.array(single_subsamp_idx), :]
        # try:
        S = empirical_covariance(sub_sample)
        # except Exception as e:
        #     print(f"Error in computing empirical covariance: {e}", file=sys.stderr)
        #     traceback.print_exc(file=sys.stderr)

        # Number of observations
        nobs = sub_sample.shape[0]

        # Penalty matrix (adapt this to your actual penalty matrix logic)
        penalty_matrix = lambdax * np.ones((p,p)) # prior_matrix

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
                edge_counts = (np.abs(precision_matrix) > 1e-5).astype(int)
                return edge_counts, precision_matrix, 1
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr, flush=True)
            return np.zeros((p, p)), np.zeros((p, p)), 0



    def run_subsample_optimization(self, lambda_range):
        """
        Run optimization on the subsamples for the entire lambda range.
        """
        edge_counts_all = np.zeros((self.p, self.p, len(lambda_range)))
        success_counts = np.zeros(len(lambda_range))

        # Replace this loop with calls to your actual optimization routine
        for q_idx in tqdm(self.subsample_indices):
            for lambdax in lambda_range:
                edge_counts, precision_matrix, success_check = self.optimize_for_q_and_j(q_idx, lambdax)
                l_idx = np.where(lambda_range == lambdax)[0][0]
                edge_counts_all[:, :, l_idx] += edge_counts
                success_counts[l_idx] += success_check

        return edge_counts_all, success_counts

def STRING_adjacency_matrix(nodes_df, edges_df):
    """
    Generate an adjacency matrix from the edgelist and nodelist obtained from STRING database. 
    """
    # Mapping Ensembl IDs to 'query term' names
    id_to_query_term = pd.Series(nodes_df['query term'].values, index=nodes_df['name']).to_dict()

    # Create a unique list of 'query terms'
    unique_query_terms = nodes_df['query term'].unique()

    # Initialize an empty adjacency matrix with unique query term labels
    adjacency_matrix = pd.DataFrame(0, index=unique_query_terms, columns=unique_query_terms)

    # Process each edge in the edges file
    for _, row in edges_df.iterrows():
        # Extract Ensembl IDs from the edge and map them to 'query term' names
        gene1_id, gene2_id = row['name'].split(' (pp) ')
        gene1_query_term = id_to_query_term.get(gene1_id)
        gene2_query_term = id_to_query_term.get(gene2_id)

        # Check if both gene names (query terms) are in the list of unique query terms
        if gene1_query_term in unique_query_terms and gene2_query_term in unique_query_terms:
            # Set the undirected edge in the adjacency matrix
            adjacency_matrix.loc[gene1_query_term, gene2_query_term] = 1
            adjacency_matrix.loc[gene2_query_term, gene1_query_term] = 1


    return adjacency_matrix

def mainpig(p, n, Q, llo, lhi, lamlen, b_perc, fp_fn, skew, synth_density, prior_conf, seed, run_type, cms, rank, size, machine='local'):
    #######################
    b = int(b_perc * n)   # size of sub-samples
    lambda_range = np.linspace(llo, lhi, lamlen)
    #######################

    if run_type == 'synthetic':
        # Synthetic run, using generated scale-free networks and data
        data, prior_matrix, adj_matrix = QJSweeper.generate_synth_data(p, n, fp_fn, skew, synth_density, seed=seed)
        synthetic_QJ = QJSweeper(data, prior_matrix, b, Q, rank, size)

        edge_counts_all, success_counts = synthetic_QJ.run_subsample_optimization(lambda_range)

    elif run_type == 'proteomics' or run_type == 'transcriptomics':
        # Omics run
        # Loading data
        cms_data = pd.read_csv(f'data/{run_type}_for_pig_{cms}.csv', index_col=0)
        p = cms_data.shape[1]
        cms_array = cms_data.values

        # scale and center 
        cms_array = (cms_array - cms_array.mean(axis=0)) / cms_array.std(axis=0)

        n = cms_array.shape[0]
        b = int(b_perc * n)
      

        print(f'Variables, Samples: {p, n}')

        # LOADING PRIOR
        # Loading Edges and Nodes with 90% or above confidence according to STRING
        edges_file = f'data/prior_data/RPPA_prior_EDGES{prior_conf}perc.csv'
        nodes_file = f'data/prior_data/RPPA_prior_NODES{prior_conf}perc.csv'

        if os.path.exists(edges_file) and os.path.exists(nodes_file):
            # Loading Edges and Nodes with 90% or above confidence according to STRING
            STRING_edges_df = pd.read_csv(edges_file)
            STRING_nodes_df = pd.read_csv(nodes_file)

            # Construct the adjacency matrix from STRING
            cms_omics_prior = STRING_adjacency_matrix(STRING_nodes_df, STRING_edges_df)
            prior_matrix = cms_omics_prior.values

            # write prior matrix to file
            with open(f'data/prior_data/RPPA_prior_adj{prior_conf}perc.pkl', 'wb') as f:
                pickle.dump(prior_matrix, f)
        else:
            print(f"Warning: One or both prior files not found. Using empty prior matrix.")
            # Generate an empty numpy array of the same shape as cms_array
            prior_matrix = np.zeros((cms_array.shape[1], cms_array.shape[1]))

        # run QJ Sweeper
        omics_QJ = QJSweeper(cms_array, prior_matrix, b, Q, rank, size, seed=seed)

        edge_counts_all, success_counts = omics_QJ.run_subsample_optimization(lambda_range)

    return edge_counts_all, prior_matrix

if __name__ == "__main__":
    # Initialize MPI communicator, rank, and size
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Run QJ Sweeper with command-line arguments.')
    parser.add_argument('--p', type=int, default=50, help='Number of variables (nodes)')
    parser.add_argument('--n', type=int, default=500, help='Number of samples')
    parser.add_argument('--Q', type=int, default=800, help='Number of sub-samples')
    parser.add_argument('--b_perc', type=float, default=0.7, help='Size of sub-samples (as a percentage of n)')
    parser.add_argument('--llo', type=float, default=0.01, help='Lower bound for lambda range')
    parser.add_argument('--lhi', type=float, default=0.4, help='Upper bound for lambda range')
    parser.add_argument('--lamlen', type=int, default=40, help='Number of points in lambda range')
    parser.add_argument('--run_type', type=str, default='synthetic', choices=['synthetic', 'proteomics', 'transcriptomics'], help='Type of run to execute')
    parser.add_argument('--prior_conf', type=str, default=90, help='Confidence level of STRING prior')
    parser.add_argument('--cms', type=str, default='cmsALL', choices=['cmsALL', 'cms123'], help='CMS type to run for omics run')
    parser.add_argument('--fp_fn', type=float, default=0, help='Chance of getting a false negative or a false positive')
    parser.add_argument('--skew', type=float, default=0, help='Skewness of the data')
    parser.add_argument('--synth_density', type=float, default=0.03, help='Density of the synthetic network')
    parser.add_argument('--seed', type=int, default=42, help='Seed for generating synthetic data')

    args = parser.parse_args()

    p,n,Q = args.p, args.n, args.Q

    # Check if running in SLURM environment
    if "SLURM_JOB_ID" in os.environ:
        edge_counts, prior_matrix = mainpig(p=args.p,
                                                  n=args.n,
                                                  Q=args.Q,
                                                  llo=args.llo,
                                                  lhi=args.lhi,
                                                  lamlen=args.lamlen,
                                                  b_perc=args.b_perc,
                                                  fp_fn=args.fp_fn,
                                                  skew=args.skew,
                                                  synth_density=args.synth_density,
                                                  prior_conf=args.prior_conf,
                                                  seed=args.seed,
                                                  run_type=args.run_type,
                                                  cms=args.cms, 
                                                  rank=rank, 
                                                  size=size, 
                                                  machine='hpc')

        num_elements = p * p * args.lamlen
        sendcounts = np.array([num_elements] * size)
        displacements = np.arange(size) * num_elements

        if rank == 0:
            # Gather the results at the root
            all_edges = np.empty(size * num_elements, dtype=edge_counts.dtype)
        else:
            all_edges = None

        comm.Gatherv(sendbuf=edge_counts.flatten(), recvbuf=(all_edges, sendcounts, displacements, MPI.DOUBLE), root=0)

        if rank == 0:
            # Reshape all_edges back to the original shape (size, p, p, len(lambda_range))
            reshaped_edges = all_edges.reshape(size, p, p, args.lamlen)

            combined_edge_counts = np.sum(reshaped_edges, axis=0)


            # Save combined results
            with open(f'results/net_results/{args.run_type}_{args.cms}_edge_counts_all_pnQ{args.p}_{args.n}_{args.Q}_{args.llo}_{args.lhi}_ll{args.lamlen}_b{args.b_perc}_fpfn{args.fp_fn}_skew{args.skew}_dens{args.synth_density}_s{args.seed}.pkl', 'wb') as f:
                pickle.dump(combined_edge_counts, f)


            # Transfer results to $HOME
            os.system("cp -r results/net_results/ $HOME/MONIKA/results/")

    else:
        # If no SLURM environment, run for entire lambda range
        edge_counts, prior_matrix = mainpig(p=args.p,
                                                  n=args.n,
                                                  Q=args.Q,
                                                  llo=args.llo,
                                                  lhi=args.lhi,
                                                  lamlen=args.lamlen,
                                                  b_perc=args.b_perc,
                                                  fp_fn=args.fp_fn,
                                                  skew=args.skew,
                                                  synth_density=args.synth_density,
                                                  prior_conf=args.prior_conf,
                                                  seed=args.seed,
                                                  run_type=args.run_type,
                                                  cms=args.cms, 
                                                  rank=1, 
                                                  size=1, 
                                                  machine='local')

        # Save results to a pickle file
        with open(f'results/net_results/{args.run_type}_{args.cms}_edge_counts_all_pnQ{p}_{args.n}_{args.Q}_{args.llo}_{args.lhi}_ll{args.lamlen}_b{args.b_perc}_fpfn{args.fp_fn}_dens{args.synth_density}_s{args.seed}.pkl', 'wb') as f:
            pickle.dump(edge_counts, f)


# scp mbarylli@snellius.surf.nl:"phase_1_code/Networks/net_results/omics_edge_counts_all_pnQ\(100\,\ 106\,\ 300\).pkl" net_results/
 