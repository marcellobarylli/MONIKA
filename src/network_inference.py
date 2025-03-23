from estimate_lambdas import (
    estimate_lambda_np,
    estimate_lambda_wp,
    find_all_knee_points,
)
from piglasso import QJSweeper, STRING_adjacency_matrix, mainpig
from evaluation_of_graph import optimize_graph, evaluate_reconstruction

import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import scipy.stats as stats
from collections import Counter

from collections import defaultdict
import os
import sys

# from tqdm import tqdm
import tqdm
from multiprocessing import Pool
from itertools import product

from scipy.interpolate import interp1d


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analysis script with various options."
    )
    # synthetic args
    parser.add_argument(
        "--synthetic", action="store_true", help="Set program type to synthetic"
    )
    parser.add_argument(
        "--run_synth",
        action="store_true",
        help="Run the main analysis for synthetic data",
    )
    parser.add_argument(
        "--post_process",
        action="store_true",
        help="Run post-processing on synthetic run",
    )
    parser.add_argument(
        "--plot_synth",
        action="store_true",
        help="Generate plots for analysing the synthetic runs.",
    )
    parser.add_argument(
        "--synth_density",
        type=float,
        default=0.03,
        help="Density of the synthetic network",
    )
    # omics argsþ
    parser.add_argument(
        "--omics", action="store_false", help="Set program type to omics "
    )
    parser.add_argument(
        "--p", type=int, default=154, help="Number of variables (nodes)"
    )
    parser.add_argument("--n", type=int, default=1337, help="Number of samples")
    parser.add_argument("--Q", type=int, default=1000, help="Number of sub-samples")
    parser.add_argument(
        "--b_perc",
        type=float,
        default=0.65,
        help="Size of sub-samples (as a percentage of n)",
    )
    parser.add_argument(
        "--llo", type=float, default=0.01, help="Lower bound for lambda range"
    )
    parser.add_argument(
        "--lhi", type=float, default=1.5, help="Upper bound for lambda range"
    )
    parser.add_argument(
        "--lamlen", type=int, default=500, help="Number of points in lambda range"
    )
    parser.add_argument(
        "--prior_conf", type=str, default=90, help="Confidence level of STRING prior"
    )
    parser.add_argument(
        "--fp_fn",
        type=float,
        default=0,
        help="Chance of getting a false negative or a false positive",
    )
    parser.add_argument("--skew", type=float, default=0, help="Skewness of the data")
    parser.add_argument(
        "--end_slice_analysis",
        action="store_true",
        help="Determine optimal lambda range for achieving biologically accurate density.",
    )
    parser.add_argument(
        "--net_dens",
        type=str,
        default="low_dens",
        help="Set high or low network density for the omics data.",
    )
    parser.add_argument(
        "--plot_omics",
        action="store_true",
        help="Generate plots for analysing the omics runs.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for generating synthetic data"
    )

    return parser.parse_args()


os.makedirs("results/net_results/inferred_adjacencies", exist_ok=True)


if not "SLURM_JOB_ID" in os.environ:
    # Figure export settings
    from mpl_toolkits.axes_grid1 import ImageGrid

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update(
        {
            "font.size": 15,
            "figure.dpi": 100,
            "grid.alpha": 0.3,
            "axes.grid": True,
            "axes.axisbelow": True,
            "figure.figsize": (8, 6),
            "mathtext.fontset": "cm",
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.labelsize": 16,
            "legend.fontsize": 13.5,
        }
    )
    plt.rc("text", usetex=False)
    plt.rc("font", family="serif")
# original_stdout = sys.stdout
# sys.stdout = open('results/net_results/Piglasso_Logs.txt', 'w')


def analysis(
    data,
    prior_matrix,
    p,
    n,
    Q,
    lambda_range,
    lowerbound,
    upperbound,
    lamlen,
    edge_counts_all,
    prior_bool=False,
    adj_matrix=None,
    run_type="SYNTHETIC",
    omics_type="",
    plot=False,
    verbose=False,
):
    """
    Runs the analysis for synthetic data or omics data. Refer to Manuscript section 3.3.2: Prior-Incorporation Graphical LASSO (PIGLASSO) for background information.

    params:
        data: multivariate normal data to be analysed, either synthetic or omics.
        prior_matrix: prior network structure, corresponding to ground truth network with varying accuracy. omics prior from STRING.
        p: number of variables in the data (nodes in network)
        n: number of samples in the data
        Q: number of sub-samples
        lambda_range: range of lambda values to be tested
        lowerbound: lowest lambda value
        upperbound: highest lambda value
        lamlen: number of lambda values to be tested
        edge_counts_all: edge counts for all lambda values
        prior_bool: boolean value to determine if prior matrix
        adj_matrix: ground truth network
        run_type: type of data being analysed (SYNTHETIC or OMICS)
        plot: boolean value to determine if plots are generated
        verbose: boolean value to determine if verbose output is printed
    returns:
        precision_matrix: inferred precision matrix (inferred network from data)
        edge_counts: number of edges in the inferred network
        density: density of the inferred network
        lambda_np: optimal lambda for non-prior term
        lambda_wp: optimal lambda for prior term
        evaluation_metrics: evaluation metrics for synthetic data
        tau_tr: tau parameter for omics data
    """

    # KNEE POINTS. Refer to manuscript section 3.3.2.2: Optimisation via Block Coordinate Descent for info on knee points
    (
        left_knee_point,
        main_knee_point,
        right_knee_point,
        left_knee_point_index,
        knee_point_index,
        right_knee_point_index,
    ) = find_all_knee_points(lambda_range, edge_counts_all)

    l_lo = left_knee_point_index  # Set it at knee-point index -1
    l_hi = right_knee_point_index  # set at knee-point index

    select_lambda_range = lambda_range[l_lo:l_hi]
    select_edge_counts_all = edge_counts_all[:, :, l_lo:l_hi]

    if verbose:
        custom_print(f"complete lambda range: {lowerbound, upperbound}")  # HERE
        custom_print(
            f"Selected lambda range: {select_lambda_range[0]} - {select_lambda_range[-1]} \n"
        )

    # LAMBDAS
    lambda_np, theta_mat = estimate_lambda_np(
        select_edge_counts_all, Q, select_lambda_range
    )

    if prior_bool == True:
        lambda_wp, tau_tr, mus = estimate_lambda_wp(
            select_edge_counts_all, Q, select_lambda_range, prior_matrix
        )
        # lambda_wp = 0.076
    else:
        lambda_wp = 0
        tau_tr = 1e5

    if verbose:
        custom_print("lambda_np: ", lambda_np)
        custom_print("lambda_wp: ", lambda_wp, "\n")

    # GRAPH OPTIMIZATION WITH FOUND LAMBDAS
    precision_matrix, edge_counts, density = optimize_graph(
        data, prior_matrix, lambda_np, lambda_wp, verbose=verbose
    )

    if verbose:
        complete_graph_edges = (p * (p - 1)) / 2
        custom_print(
            f"Number of prior edges (lower triangular): {np.sum(prior_matrix != 0) / 2}"
        )
        custom_print(
            f"Density of prior penalty matrix: {((np.sum(prior_matrix != 0) / 2) / complete_graph_edges)}\n"
        )

        custom_print(
            "Number of edges of inferred network (lower triangular): ", edge_counts
        )
        custom_print("Density of inferred network: ", density)

    if plot == True:
        scalar_edges = np.sum(edge_counts_all, axis=(0, 1)) / (2 * Q)
        scalar_select_edges = np.sum(select_edge_counts_all, axis=(0, 1)) / (2 * Q)

        # PLOTTING JUST THE TOTAL (WITHOUT RED)
        plt.figure(figsize=(8, 6), dpi=300)
        plt.scatter(lambda_range, scalar_edges, color="grey", alpha=0.5)
        plt.scatter(select_lambda_range, scalar_select_edges, color="red", alpha=0.8)
        plt.title(rf"Edge Counts vs $\lambda$, {run_type}, {omics_type}")
        plt.xlabel(r"$ \lambda$", fontsize=15)
        plt.ylabel("Edge Counts", fontsize=12)
        # plt.ylim(0, 8000)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()

    if run_type == "SYNTHETIC":
        # RECONSTRUCTION
        evaluation_metrics = evaluate_reconstruction(adj_matrix, precision_matrix)
        if verbose:
            print("Evaluation metrics: ", evaluation_metrics)
            print("\n\n\n")

        return (
            precision_matrix,
            edge_counts,
            density,
            lambda_np,
            lambda_wp,
            evaluation_metrics,
            tau_tr,
        )

    return precision_matrix, edge_counts, density, lambda_np, lambda_wp, tau_tr


rank = 1
size = 1

if __name__ == "__main__":
    args = parse_arguments()
    ################################################## OMICS DATA PART #################################################
    # argparse arg for omics data
    if args.omics == True:
        # Preparing results file
        with open(
            "results/net_results/omics_networks_info.txt", "w", encoding="utf-8"
        ) as f_out:

            def custom_print(*args, **kwargs):
                # Convert all arguments to strings
                string_args = [str(arg) for arg in args]
                # Join the strings with spaces
                output = " ".join(string_args)
                # Write to file
                f_out.write(output + "\n")
                # Write to console
                print(output)

            for o_t in ["p", "t"]:
                for cms in ["cmsALL", "cms123"]:
                    # Parameters, remain fixed for omics data
                    p = args.p  # number of nodes (genes) in the processed dataset
                    b_perc = (
                        args.b_perc
                    )  # fixed b_perc, optimal value determined from synthetic experiments
                    n = 1337  # not actual samples, just filename requirements
                    Q = args.Q  # number of sub-samples

                    lowerbound = args.llo
                    upperbound = args.lhi
                    lamlen = args.lamlen
                    lambda_range = np.linspace(lowerbound, upperbound, lamlen)

                    fp_fn = args.fp_fn
                    skew = args.skew
                    synth_density = args.synth_density
                    prior_conf = args.prior_conf
                    seed = args.seed

                    man = False  # ALWAYS FALSE

                    if o_t == "p":
                        prior_bool = True
                        omics_type = "proteomics"
                    elif o_t == "t":
                        prior_bool = True
                        omics_type = "transcriptomics"

                    # Load omics edge counts
                    file_ = f"results/net_results/{omics_type}_{cms}_edge_counts_all_pnQ{p}_{n}_{Q}_{lowerbound}_{upperbound}_ll{lamlen}_b{b_perc}_fpfn{fp_fn}_skew{skew}_dens{synth_density}_s{seed}.pkl"

                    # Check if the file already exists
                    if os.path.exists(file_):
                        print(f"Loading existing edge counts from {file_}")
                        with open(file_, "rb") as f:
                            omics_edge_counts_all = pickle.load(f)
                    else:
                        print(
                            f"File containing edge counts not found. Running PIGLASSO to generate edge counts."
                        )
                        # Get edge counts for network inference from PIGLASSO
                        omics_edge_counts_all, _ = mainpig(
                            p=p,
                            n=n,
                            Q=Q,
                            llo=lowerbound,
                            lhi=upperbound,
                            lamlen=lamlen,
                            b_perc=b_perc,
                            fp_fn=fp_fn,
                            skew=skew,
                            synth_density=synth_density,
                            prior_conf=prior_conf,
                            seed=seed,
                            run_type=omics_type,
                            cms=cms,
                            rank=1,
                            size=1,
                            machine="local",
                        )

                        # write the file
                        with open(file_, "wb") as f:
                            pickle.dump(omics_edge_counts_all, f)

                    # Load Omics Data
                    cms_data = pd.read_csv(
                        f"data/{omics_type}_for_pig_{cms}.csv", index_col=0
                    )
                    cms_array = cms_data.values

                    # Load STRING prior edges and nodes
                    STRING_edges_df = pd.read_csv(
                        f"data/prior_data/RPPA_prior_EDGES{prior_conf}perc.csv"
                    )
                    STRING_nodes_df = pd.read_csv(
                        f"data/prior_data/RPPA_prior_NODES{prior_conf}perc.csv"
                    )

                    # # Construct the adjacency matrix from STRING
                    cms_omics_prior = STRING_adjacency_matrix(
                        STRING_nodes_df, STRING_edges_df
                    )
                    # write prior matrix to file
                    with open(
                        f"data/prior_data/RPPA_prior_adj{prior_conf}perc.pkl", "wb"
                    ) as f:
                        pickle.dump(cms_omics_prior, f)

                    # print(f'---------------------------------------------------got dat RPPA_prior_adj{prior_conf}perc.pkl')

                    if prior_bool == True:
                        # print density of prior
                        complete_g = p * (p - 1)
                        prior_density = np.sum(cms_omics_prior.values) / complete_g
                    else:
                        # only keep columns / rows that are in the omics data
                        cms_omics_prior = cms_omics_prior[cms_data.columns]
                        cms_omics_prior = cms_omics_prior.reindex(
                            index=cms_data.columns
                        )
                        cms_omics_prior = cms_omics_prior * 0

                    cms_omics_prior_matrix = (
                        cms_omics_prior.values * 0.9
                    )  # Adjust prior confidence to that of STRING

                    p = cms_array.shape[
                        1
                    ]  # number of nodes (genes) in the processed dataset
                    n = cms_array.shape[0]  # number of samples in the processed dataset
                    b = int(0.65 * n)  # percentage of total samples in sub-sample

                    # scale and center
                    cms_array = (cms_array - cms_array.mean(axis=0)) / cms_array.std(
                        axis=0
                    )

                    custom_print(
                        f"--------------------------------------\n{str.upper(omics_type)}, {cms} RESULTS\n--------------------------------------\n",
                        file=f,
                    )

                    custom_print(f"Number of samples: {n}")
                    custom_print(f"Number of sub-samples: {Q}")
                    custom_print(f"Number of variables: {p}\n")

                    # Default program run
                    if not args.end_slice_analysis == True:
                        if args.net_dens == "high_dens":  # high density network
                            end_slice = 325
                        else:
                            end_slice = 250  # low density network, closer to prior

                        sliced_omics_edge_counts_all = omics_edge_counts_all[
                            :, :, :-end_slice
                        ]

                        # SETTING LAMBDA DIMENSIONS TO FIT THE DATA
                        new_lamlen = sliced_omics_edge_counts_all.shape[2]
                        new_upperbound = lowerbound + (upperbound - lowerbound) * (
                            new_lamlen - 1
                        ) / (lamlen - 1)
                        lambda_range = np.linspace(
                            lowerbound, new_upperbound, new_lamlen
                        )

                        (
                            precision_mat,
                            edge_counts,
                            density,
                            lambda_np,
                            lambda_wp,
                            tau_tr,
                        ) = analysis(
                            cms_array,
                            cms_omics_prior_matrix,
                            p,
                            n,
                            Q,
                            lambda_range,
                            lowerbound,
                            new_upperbound,
                            new_lamlen,
                            sliced_omics_edge_counts_all,
                            prior_bool,
                            run_type="OMICS",
                            omics_type=f"{str.upper(omics_type)}, {cms}",
                            plot=args.plot_omics,
                            verbose=True,
                        )

                    # # The density of the final network is affected by the range of lambda values we consider.
                    # # RUN ANALYSIS for multiple END SLICES. end slice value determines the range of lambda values we consider (higher = fewer lambda values)
                    else:
                        densities = []
                        np_lams = []
                        wp_lams = []
                        tau_trs = []
                        no_end_slices = 400
                        slicer_range = range(200, no_end_slices)
                        x_axis = []
                        i = 0
                        for end_slice in slicer_range:
                            i += 1
                            sliced_omics_edge_counts_all = omics_edge_counts_all[
                                :, :, :-end_slice
                            ]

                            # SETTING LAMBDA DIMENSIONS TO FIT THE DATA
                            new_lamlen = sliced_omics_edge_counts_all.shape[2]
                            new_upperbound = lowerbound + (upperbound - lowerbound) * (
                                new_lamlen - 1
                            ) / (lamlen - 1)
                            # x_axis.append(new_upperbound)
                            x_axis.append(end_slice)

                            lambda_range = np.linspace(
                                lowerbound, new_upperbound, new_lamlen
                            )
                            (
                                precision_mat,
                                edge_counts,
                                density,
                                lambda_np,
                                lambda_wp,
                                tau_tr,
                            ) = analysis(
                                cms_array,
                                cms_omics_prior_matrix,
                                p,
                                n,
                                Q,
                                lambda_range,
                                lowerbound,
                                new_upperbound,
                                new_lamlen,
                                sliced_omics_edge_counts_all,
                                prior_bool,
                                run_type="OMICS",
                                plot=args.plot_omics,
                                verbose=False,
                            )

                            print(i, new_upperbound, o_t, cms)
                            print(
                                f"lambda_np: {lambda_np}, lambda_wp: {lambda_wp}, density: {density}"
                            )
                            densities.append(density)
                            np_lams.append(lambda_np)
                            wp_lams.append(lambda_wp)
                            tau_trs.append(tau_tr)

                        # write densities to file
                        with open(
                            f"results/net_results/endslice_densities_{omics_type}_{cms}_Q{Q}_prior{prior_bool}_slices{len(slicer_range)}.pkl",
                            "wb",
                        ) as f:
                            pickle.dump(densities, f)
                        # write np_lams to file
                        with open(
                            f"results/net_results/endslice_np_lams_{omics_type}_{cms}_Q{Q}_prior{prior_bool}_slices{len(slicer_range)}.pkl",
                            "wb",
                        ) as f:
                            pickle.dump(np_lams, f)
                        # write wp_lams to file
                        with open(
                            f"results/net_results/endslice_wp_lams_{omics_type}_{cms}_Q{Q}_prior{prior_bool}_slices{len(slicer_range)}.pkl",
                            "wb",
                        ) as f:
                            pickle.dump(wp_lams, f)
                        # write tau_trs to file
                        with open(
                            f"results/net_results/endslice_tau_trs_{omics_type}_{cms}_Q{Q}_prior{prior_bool}_slices{len(slicer_range)}.pkl",
                            "wb",
                        ) as f:
                            pickle.dump(tau_trs, f)

                        # # #

                        # Load np_lams and wp_lams from file
                        with open(
                            f"results/net_results/endslice_np_lams_{omics_type}_{cms}_Q{Q}_prior{prior_bool}_slices{len(slicer_range)}.pkl",
                            "rb",
                        ) as f:
                            np_lams = pickle.load(f)
                        with open(
                            f"results/net_results/endslice_wp_lams_{omics_type}_{cms}_Q{Q}_prior{prior_bool}_slices{len(slicer_range)}.pkl",
                            "rb",
                        ) as f:
                            wp_lams = pickle.load(f)

                        # Load tau_trs and densities from file
                        with open(
                            f"results/net_results/endslice_tau_trs_{omics_type}_{cms}_Q{Q}_prior{prior_bool}_slices{len(slicer_range)}.pkl",
                            "rb",
                        ) as f:
                            tau_trs = pickle.load(f)
                        with open(
                            f"results/net_results/endslice_densities_{omics_type}_{cms}_Q{Q}_prior{prior_bool}_slices{len(slicer_range)}.pkl",
                            "rb",
                        ) as f:
                            densities = pickle.load(f)

                        # Create a figure with 3 subplots
                        plt.figure(figsize=(18, 5))  # Adjust the size as needed

                        # First subplot for np_lams and wp_lams
                        plt.subplot(1, 3, 1)  # 1 row, 3 columns, first subplot
                        plt.plot(
                            x_axis,
                            np_lams,
                            color="red",
                            alpha=0.8,
                            label=r"$\lambda_{np}$",
                        )
                        plt.scatter(x_axis, np_lams, color="red", alpha=0.8)
                        plt.plot(
                            x_axis,
                            wp_lams,
                            color="blue",
                            alpha=0.8,
                            label=r"$\lambda_{wp}$",
                        )
                        plt.scatter(x_axis, wp_lams, color="blue", alpha=0.8)
                        plt.title(
                            f"$\lambda_np$ and $\lambda_wp$ vs end slice value for {omics_type} data, Q = {Q}"
                        )
                        plt.xlabel("End slice value", fontsize=12)
                        plt.ylabel(r"$\lambda$", fontsize=12)
                        plt.legend()
                        plt.grid()

                        # Second subplot for tau_trs
                        plt.subplot(1, 3, 2)  # 1 row, 3 columns, second subplot
                        plt.plot(x_axis, tau_trs, color="purple", alpha=0.65)
                        plt.scatter(x_axis, tau_trs, color="purple", alpha=0.65)
                        plt.title(
                            rf"$\tau_{{tr}}$ vs end slice value for {omics_type} data, Q = {Q}"
                        )
                        plt.xlabel("End slice value", fontsize=12)
                        plt.ylabel(r"$\tau_{tr}$", fontsize=12)
                        plt.grid()

                        # Third subplot for densities
                        plt.subplot(1, 3, 3)  # 1 row, 3 columns, third subplot
                        plt.plot(x_axis, densities, color="red", alpha=0.8)
                        plt.scatter(x_axis, densities, color="red", alpha=0.8)
                        plt.title(
                            f"Density vs end slice value for {omics_type} data, Q = {Q}"
                        )
                        plt.xlabel("End slice value", fontsize=12)
                        plt.ylabel("Density", fontsize=12)
                        plt.grid()

                        plt.tight_layout()

                        # Show the figure
                        plt.show()

                    # print tau_tr value
                    custom_print(f"\ntau_tr: {tau_tr}")

                    # get adjacency from precision matrix
                    adj_matrix = (np.abs(precision_mat) > 1e-5).astype(int)
                    # assign columns and indices of prior matrix to adj_matrix
                    adj_matrix = pd.DataFrame(
                        adj_matrix, index=cms_data.columns, columns=cms_data.columns
                    )

                    # save inferred network as adjacency matrix
                    adj_matrix.to_csv(
                        f"results/net_results/inferred_adjacencies/{omics_type}_{cms}_adj_matrix_p{p}_Lambda_np{not man}_{args.net_dens}.csv"
                    )
            print(
                "Finished saving adjacency matrices for all omics layers and cms types (aggressive and non-mesenchymal).\n\n"
            )

            proteomics_ALL_net = pd.read_csv(
                f"results/net_results/inferred_adjacencies/proteomics_cmsALL_adj_matrix_p154_Lambda_np{not man}_{args.net_dens}.csv",
                index_col=0,
            )
            transcriptomics_ALL_net = pd.read_csv(
                f"results/net_results/inferred_adjacencies/transcriptomics_cmsALL_adj_matrix_p154_Lambda_np{not man}_{args.net_dens}.csv",
                index_col=0,
            )
            proteomics_123_net = pd.read_csv(
                f"results/net_results/inferred_adjacencies/proteomics_cms123_adj_matrix_p154_Lambda_np{not man}_{args.net_dens}.csv",
                index_col=0,
            )
            transcriptomics_123_net = pd.read_csv(
                f"results/net_results/inferred_adjacencies/transcriptomics_cms123_adj_matrix_p154_Lambda_np{not man}_{args.net_dens}.csv",
                index_col=0,
            )

            # compare similarity of all networks to each other
            proteomics_ALL_net = proteomics_ALL_net.values
            transcriptomics_ALL_net = transcriptomics_ALL_net.values
            proteomics_123_net = proteomics_123_net.values
            transcriptomics_123_net = transcriptomics_123_net.values

            def print_comparison(net1_name, net2_name, similarity_dict):
                print(f"\nComparing {net1_name} to {net2_name}:")
                for key, value in similarity_dict.items():
                    print(f"  {key}: {value}")

            custom_print("--------------------------------------")
            custom_print("COMPARING ALL INFERRED NETWORKS TO EACH OTHER")
            custom_print("--------------------------------------")

            comparisons = [
                ("proteomics_ALL_net", "transcriptomics_ALL_net"),
                ("proteomics_ALL_net", "proteomics_123_net"),
                ("proteomics_ALL_net", "transcriptomics_123_net"),
                ("transcriptomics_ALL_net", "proteomics_123_net"),
                ("transcriptomics_ALL_net", "transcriptomics_123_net"),
                ("proteomics_123_net", "transcriptomics_123_net"),
            ]

            prior_comparisons = [
                ("proteomics_ALL_net", "prior"),
                ("transcriptomics_ALL_net", "prior"),
                ("proteomics_123_net", "prior"),
                ("transcriptomics_123_net", "prior"),
            ]

            for net1, net2 in comparisons:
                similarity = evaluate_reconstruction(globals()[net1], globals()[net2])
                print_comparison(net1, net2, similarity)

            custom_print("\n--------------------------------------")
            custom_print("COMPARING INFERRED NETWORKS TO PRIOR")
            custom_print("--------------------------------------")

            # Read pkl prior file
            with open("data/prior_data/RPPA_prior_adj90perc.pkl", "rb") as f:
                prior = pickle.load(f)

            # Compare networks to prior
            for net, _ in prior_comparisons:
                similarity = evaluate_reconstruction(globals()[net], prior.values)
                print_comparison(net, "prior", similarity)

    # ################################################# SYNTHETIC PART #################################################
    # argparse arg for synthetic
    if args.synthetic:
        # COMPLETE SWEEP
        p_values = [150]
        n_values = [50, 100, 300, 500, 700, 900, 1100]
        b_perc_values = [0.6, 0.65, 0.7]
        fp_fn_values = [0.0, 0.6, 0.8, 1]
        seed_values = list(range(1, 31))
        dens_values = [0.05]
        man_values = [False]

        # Fixed parameters
        Q = 1000
        llo = 0.01
        lhi = 0.5
        lamlen = 100
        skew = 0
        prior_bool = True

        lambda_range = np.linspace(llo, lhi, lamlen)

        # Initialize a dictionary to hold f1 scores for averaging
        f1_scores = {}
        recall_scores = {}

        if args.run_synth == True:

            def worker_function(params):
                """
                Performs a Hypercube analysis for a given set of parameters.
                params:
                    p: Number of nodes in network
                    n: Number of samples
                    b_perc: Percentage of samples in sub-sample
                    fp_fn: False positive/false negative rate
                    seed: Random seed for reproducible data generation
                    dens: Density of the network
                    man: Manual lambda parameter (ALWAYS FALSE)
                """
                p, n, b_perc, fp_fn, seed, dens, man = params

                # Fixed parameters
                Q = 1000
                llo = 0.01
                lhi = 0.5
                lamlen = 100
                skew = 0
                prior_bool = True
                lambda_range = np.linspace(llo, lhi, lamlen)

                # Calculate the size of sub-samples (b)
                b = int(b_perc * n)

                # Construct filename for edge counts
                filename_edges = f"results/net_results/synthetic_cmsALL_edge_counts_all_pnQ{p}_{n}_{Q}_{llo}_{lhi}_ll{lamlen}_b{b_perc}_fpfn0.0_skew0_dens{dens}_s{seed}.pkl"
                param_key = (p, n, b_perc, fp_fn, seed, dens, str(man))

                if not os.path.isfile(filename_edges):
                    return None  # File does not exist

                try:
                    with open(filename_edges, "rb") as f:
                        synth_edge_counts_all = pickle.load(f)
                except EOFError:
                    print(f"Failed to load file: {filename_edges}")
                    return None  # Skip this file and return

                # Process the edge counts
                synth_edge_counts_all = synth_edge_counts_all  #  / (2 * Q)

                synth_data, synth_prior_matrix, synth_adj_matrix = (
                    QJSweeper.generate_synth_data(
                        p,
                        n,
                        skew=skew,
                        fp_fn_chance=fp_fn,
                        synth_density=dens,
                        seed=seed,
                    )
                )

                overlap = np.sum(
                    (synth_prior_matrix == 1) & (synth_adj_matrix == 1)
                ) / (np.sum(synth_prior_matrix == 1))

                if fp_fn == 1:
                    synth_prior_matrix = synth_prior_matrix * 0
                    prior_bool = False

                # Run analysis
                # Currently, analysis including lambda range plotting only if full sweep is done.
                _, _, _, lambda_np, lambda_wp, temp_evalu, tau_tr = analysis(
                    synth_data,
                    synth_prior_matrix,
                    p,
                    n,
                    Q,
                    lambda_range,
                    llo,
                    lhi,
                    lamlen,
                    synth_edge_counts_all,
                    prior_bool=prior_bool,
                    adj_matrix=synth_adj_matrix,
                    run_type="SYNTHETIC",
                    plot=args.plot_synth,
                    verbose=False,
                )

                return {
                    "param_key": param_key,
                    "f1_score": temp_evalu["f1_score"],
                    "precision": temp_evalu["precision"],
                    "recall": temp_evalu["recall"],
                    "jaccard_similarity": temp_evalu["jaccard_similarity"],
                    "overlap": overlap,
                    "tau_tr": tau_tr,
                    "lambda_np": lambda_np,
                    "lambda_wp": lambda_wp,
                }

            def update_progress(*a):
                pbar.update()

            # if __name__ == "__main__":
            parameter_combinations = list(
                product(
                    p_values,
                    n_values,
                    b_perc_values,
                    fp_fn_values,
                    seed_values,
                    dens_values,
                    man_values,
                )
            )

            with Pool() as pool:
                pbar = tqdm.tqdm(total=len(parameter_combinations))
                results = [
                    pool.apply_async(
                        worker_function, args=(params,), callback=update_progress
                    )
                    for params in parameter_combinations
                ]

                # Close the pool and wait for each task to complete
                pool.close()
                pool.join()
                pbar.close()

            # Extract the results from each async result object
            results = [res.get() for res in results]

            # Organize results
            organized_results = {
                result["param_key"]: {
                    "f1_score": result["f1_score"],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "jaccard_similarity": result["jaccard_similarity"],
                    "overlap": result["overlap"],
                    "tau_tr": result["tau_tr"],
                    "lambda_np": result["lambda_np"],
                    "lambda_wp": result["lambda_wp"],
                }
                for result in results
                if result is not None
            }

            # save to file
            with open(
                f"results/net_results/sweep_results/organized_SWEEP_results_n{len(n_values)}_withjaccetc1000.pkl",
                "wb",
            ) as f:
                pickle.dump(organized_results, f)

            print("Organized synthetic results saved.")

        # Initialize dictionaries for average scores and SDs
        average_scores = {
            "f1_score": {},
            "precision": {},
            "recall": {},
            "jaccard_similarity": {},
            "overlap": {},
            "tau_tr": {},
            "lambda_np": {},
            "lambda_wp": {},
        }
        SD_scores = {
            "f1_score": {},
            "precision": {},
            "recall": {},
            "jaccard_similarity": {},
            "overlap": {},
            "tau_tr": {},
            "lambda_np": {},
            "lambda_wp": {},
        }

        # Loop over parameter combinations
        for p in p_values:
            for n in n_values:
                for b_perc in b_perc_values:
                    for fp_fn in fp_fn_values:
                        for man in [str(man) for man in man_values]:
                            # Initialize lists for each score
                            scores_for_average = {
                                "f1_score": [],
                                "precision": [],
                                "recall": [],
                                "jaccard_similarity": [],
                                "overlap": [],
                                "tau_tr": [],
                                "lambda_np": [],
                                "lambda_wp": [],
                            }

                            # New key without seed and dens
                            new_key = (p, n, b_perc, fp_fn, man)

                            # Loop over seeds and densities
                            for seed in seed_values:
                                for dens in dens_values:
                                    key = (p, n, b_perc, fp_fn, seed, dens, man)
                                    result = organized_results.get(key)
                                    if result:  # Check if the result exists
                                        for metric in scores_for_average.keys():
                                            scores_for_average[metric].append(
                                                result[metric]
                                            )

                            # Calculating the average and SD for each metric
                            for metric in scores_for_average.keys():
                                if scores_for_average[metric]:
                                    average_scores[metric][new_key] = np.mean(
                                        scores_for_average[metric]
                                    )
                                    SD_scores[metric][new_key] = np.std(
                                        scores_for_average[metric], ddof=1
                                    )  # Use ddof=1 for sample standard deviation
                                else:
                                    # Handle missing data
                                    average_scores[metric][new_key] = None
                                    SD_scores[metric][new_key] = None

        # Save average scores to files
        metrics = [
            "f1_score",
            "precision",
            "recall",
            "jaccard_similarity",
            "tau_tr",
            "lambda_np",
            "lambda_wp",
            "overlap",
        ]

        for metric in metrics:
            with open(
                f"results/net_results/sweep_results/average_{metric}_scores.pkl", "wb"
            ) as f:
                pickle.dump(average_scores[metric], f)

            with open(
                f"results/net_results/sweep_results/SD_{metric}_scores.pkl", "wb"
            ) as f:
                pickle.dump(SD_scores[metric], f)

        metrics = [
            "f1_score",
            "precision",
            "recall",
            "jaccard_similarity",
            "tau_tr",
            "lambda_np",
            "lambda_wp",
            "overlap",
        ]

        for metric in metrics:
            avg_file = f"results/net_results/sweep_results/average_{metric}_scores.pkl"
            sd_file = f"results/net_results/sweep_results/SD_{metric}_scores.pkl"

            if os.path.exists(avg_file):
                with open(avg_file, "rb") as f:
                    average_scores[metric] = pickle.load(f)
            else:
                print(
                    f"Warning: {avg_file} does not exist. Please re-run script with args --synthetic and --run_synth set to True."
                )

            if os.path.exists(sd_file):
                with open(sd_file, "rb") as f:
                    SD_scores[metric] = pickle.load(f)
            else:
                print(
                    f"Warning: {sd_file} does not exist. Please re-run script with args --synthetic and --run_synth.."
                )

        # argparse for plotting
        if args.plot_synth:
            # Plotting validation results for varying parameter setting of b_perc
            n = 700  # Fixed sample size
            p = 150  # Fixed number of variables
            man_values = [False]

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))  # 1x2 subplot

            for fp_fn in fp_fn_values:
                f1_scores = []
                recall_scores = []
                f1_errors = []
                recall_errors = []
                overlap_values = []

                for b_perc in b_perc_values:
                    key = (p, n, b_perc, fp_fn, str(man_values[0]))
                    f1_scores.append(average_scores["f1_score"].get(key))
                    recall_scores.append(average_scores["recall"].get(key))
                    f1_errors.append(
                        SD_scores["f1_score"].get(key, 0)
                    )  # Default to 0 if no SD available
                    recall_errors.append(
                        SD_scores["recall"].get(key, 0)
                    )  # Default to 0 if no SD available
                    overlap_values.append(average_scores["overlap"].get(key, 0))

            if None in overlap_values:
                print(
                    "Error: None values found in sweep results. Please perform synth sweep on HPC before running synth experiments again."
                )
            else:
                for b_perc in b_perc_values:
                    # Determine color based on fp_fn value
                    color = plt.cm.viridis(fp_fn / max(fp_fn_values))

                    # Plot F1 scores
                    axes[0].errorbar(
                        b_perc_values,
                        f1_scores,
                        yerr=f1_errors,
                        label=f"fp_fn={fp_fn}, avg overlap={np.mean(overlap_values):.2f}",
                        fmt="-o",
                        color=color,
                    )
                    axes[0].set_title("F1 Scores vs b_perc")
                    axes[0].set_xlabel("b_perc")
                    axes[0].set_ylabel("F1 Score")
                    axes[0].legend(loc="best", fontsize="small")
                    axes[0].grid(alpha=0.3)

                    # Plot Recall scores
                    axes[1].errorbar(
                        b_perc_values,
                        recall_scores,
                        yerr=recall_errors,
                        label=f"fp_fn={fp_fn}, avg overlap={np.mean(overlap_values):.2f}",
                        fmt="-o",
                        color=color,
                    )
                    axes[1].set_title("Recall Scores vs b_perc")
                    axes[1].set_xlabel("b_perc")
                    axes[1].set_ylabel("Recall Score")
                    axes[1].legend(loc="best", fontsize="small")
                    axes[1].grid(alpha=0.3)

                plt.suptitle(
                    f"Performance Metrics vs b_perc (n={n}, p={p})", fontsize=16
                )
                plt.tight_layout()
                plt.savefig(f"results/net_results/sweep_results/b_perc_plot.svg")
                plt.show()

                # PLOTTING PERFORMANCE METRICS VS SAMPLE SIZE
                def reversed_colormap(cmap_name):
                    cmap = plt.cm.get_cmap(cmap_name)
                    colors = cmap(np.arange(cmap.N))
                    colors = np.flipud(colors)
                    return mcolors.LinearSegmentedColormap.from_list(
                        "reversed_" + cmap_name, colors
                    )

                reversed_blues = reversed_colormap("Blues")
                b_perc = 0.65  # Fixed b_perc
                p = 150  # Fixed number of variables
                fig, axes = plt.subplots(
                    nrows=1, ncols=2, figsize=(15, 6), sharey=True, dpi=150
                )  # 1x2 subplot
                man = False  # Only considering False for man values

                for fp_fn in fp_fn_values:
                    f1_scores = []
                    recall_scores = []
                    f1_errors = []
                    recall_errors = []
                    for n in n_values:
                        key = (p, n, b_perc, fp_fn, str(man))
                        f1_scores.append(average_scores["f1_score"].get(key))
                        recall_scores.append(average_scores["recall"].get(key))
                        f1_errors.append(
                            SD_scores["f1_score"].get(key, 0)
                        )  # Default to 0 if no SD available
                        recall_errors.append(
                            SD_scores["recall"].get(key, 0)
                        )  # Default to 0 if no SD available

                    # Determine color and alpha based on fp_fn value
                    if fp_fn < 1.0:
                        exponent = 3
                        scaled_fp_fn = fp_fn**exponent
                        color = reversed_blues(scaled_fp_fn)
                        alpha = 1
                    else:
                        color = "firebrick"
                        alpha = 1

                    # Plot F1 scores
                    axes[0].errorbar(
                        n_values,
                        f1_scores,
                        yerr=f1_errors,
                        fmt="-o",
                        color=color,
                        alpha=alpha,
                        markersize=3,
                        label=f"fp_fn={fp_fn}",
                    )
                    axes[0].set_ylabel("F-Score", fontsize=12)
                    axes[0].set_xlabel("Sample Size", fontsize=12)
                    axes[0].grid(alpha=0.15)
                    axes[0].set_title("F-Score vs Sample Size", fontsize=14)

                    # Plot Recall scores
                    axes[1].errorbar(
                        n_values,
                        recall_scores,
                        yerr=recall_errors,
                        fmt="-o",
                        color=color,
                        alpha=alpha,
                        markersize=3,
                        label=f"fp_fn={fp_fn}",
                    )
                    axes[1].set_ylabel("Recall", fontsize=12)
                    axes[1].set_xlabel("Sample Size", fontsize=12)
                    axes[1].set_title("Recall vs Sample Size", fontsize=14)
                    axes[1].grid(alpha=0.15)

                xticks = [0, 100, 300, 500, 700, 900, 1100]
                for ax in axes:  # Apply to both subplots
                    ax.set_xticks(xticks)
                    ax.set_xlim(0, 1150)
                    ax.legend(fontsize=10, loc="best")

                plt.suptitle(
                    f"Performance Metrics vs Sample Size (b_perc={b_perc}, p={p})",
                    fontsize=16,
                )  # Add overall title
                plt.tight_layout()
                plt.savefig(f"results/net_results/sweep_results/n_value_plot.svg")
                plt.show()

                # PLOTTING PRIOR OVERLAP VS TAU_TR
                # Load organized results
                with open(
                    f"results/net_results/sweep_results/organized_SWEEP_results_n{len(n_values)}.pkl",
                    "rb",
                ) as f:
                    organized_results = pickle.load(f)

                # Organize data by 'overlap', excluding cases where overlap is 0.0
                organized_data = {}
                for key, value in organized_results.items():
                    if value["overlap"] == 0.0:
                        continue  # Skip this entry if overlap is 0.0

                    overlap = value["overlap"]
                    tau_tr = value["tau_tr"]

                    if overlap not in organized_data:
                        organized_data[overlap] = []

                    organized_data[overlap].append(tau_tr)

                # Calculate mean and standard deviation for each 'overlap'
                overlap_values = np.array(list(organized_data.keys()))
                mean_tau_tr_values = np.array(
                    [np.mean(organized_data[ov]) for ov in overlap_values]
                )
                error_tau_tr_values = np.array(
                    [np.std(organized_data[ov], ddof=1) for ov in overlap_values]
                )  # ddof=1 for sample standard deviation

                # Create a linear interpolation function
                f = interp1d(overlap_values, mean_tau_tr_values, kind="linear")

                # Specific tau_tr values and their colors
                tau_tr_points = [739.5, 739.8, 751, 754]
                colors = ["red", "red", "blue", "blue"]

                # Plotting the error bars and line
                plt.figure()  # (figsize=(7, 4), dpi=300)
                plt.errorbar(
                    overlap_values,
                    mean_tau_tr_values,
                    yerr=error_tau_tr_values,
                    fmt="o",
                    color="purple",
                    alpha=0.5,
                )
                plt.plot(overlap_values, mean_tau_tr_values, color="purple", alpha=0.5)

                # Plot specific tau_tr points
                for tau_tr, color in zip(tau_tr_points, colors):
                    # Assuming a linear relationship, find the corresponding overlap value
                    corresponding_overlap = np.interp(
                        tau_tr, mean_tau_tr_values, overlap_values
                    )
                    plt.scatter(
                        corresponding_overlap, tau_tr, color=color, marker="s", s=50
                    )  # s is the size of the square

                plt.xlabel("Prior Overlap")
                plt.ylabel(r"$\tau^{tr}$", fontsize=18)
                # plt.grid(alpha=0.2)
                plt.suptitle(r"Prior Overlap vs. $\tau^{tr}$")
                plt.tight_layout()

                plt.savefig(f"results/net_results/sweep_results/tau_tr_vs_overlap.svg")
                plt.show()

# sys.stdout.close()
# sys.stdout = original_stdout
