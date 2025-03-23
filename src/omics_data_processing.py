# %%
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
import matplotlib.pyplot as plt
import requests
import pandas as pd
import glob
import zipfile
from tqdm import tqdm
import logging
from scipy import stats
import scipy.stats as stats
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import PowerTransformer
import statsmodels


def apply_yeo_johnson(column):
    """
    Applies a power transform to the data using the Yeo-Johnson method, to make it more Gaussian-like.
    """
    transformer = PowerTransformer(method="yeo-johnson")
    # Reshape data for transformation (needs to be 2D)
    column_reshaped = column.values.reshape(-1, 1)
    transformed_column = transformer.fit_transform(column_reshaped)
    # Flatten the array to 1D
    return transformed_column.flatten()


def filter_dataframes(df1, df2):
    # check overlap between df1.columns and df2.columns
    overlap = set(df1.columns).intersection(set(df2.columns))
    print("total number of variables in df2: {}".format(len(df2.columns)))
    print(f"variables both in df1 and df2: {len(overlap)}")

    # keep only columns that are in overlap
    df1 = df1.loc[:, df1.columns.isin(overlap)]
    df2 = df2.loc[:, df2.columns.isin(overlap)]

    return df1, df2


def center_and_scale(df, axis=0):
    # center and scale across columns
    df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    return df


def transform_and_test(data_to_trim, dataframe_name):
    """
    Winsorize outliers, apply Yeo-Johnson transformation, and perform Kolmogorov-Smirnoff test to check for normality.
    """
    print("--------------------------------------------------------------")
    print(f"Initial results for {dataframe_name}")

    # Winsorize
    data_to_trim = data_to_trim.apply(
        lambda x: winsorize(x, limits=[0.01, 0.01]), axis=0
    )

    alpha = 0.05
    original_ks_results = {}

    # Perform initial K-S test and store results
    for column in data_to_trim.columns:
        data = data_to_trim[column].dropna()
        if data.nunique() > 1 and len(data) > 3:
            stat, p = stats.kstest(data, "norm", args=(data.mean(), data.std()))
            original_ks_results[column] = (stat, p)
            if p < alpha:
                # Apply Yeo-Johnson transformation
                data_to_trim[column] = apply_yeo_johnson(data_to_trim[column])

    # Perform K-S test again on transformed columns and compare
    for column, (original_stat, original_p) in original_ks_results.items():
        if original_p < alpha:
            transformed_data = data_to_trim[column].dropna()
            new_stat, new_p = stats.kstest(
                transformed_data,
                "norm",
                args=(transformed_data.mean(), transformed_data.std()),
            )
            if new_p < alpha:
                print(f"Column: {column}")
                print(
                    f"  Original K-S Statistic: {original_stat}, p-value: {original_p}"
                )
                print(f"  Transformed K-S Statistic: {new_stat}, p-value: {new_p}")
                print(
                    "--------------------------------------------------------------\n"
                )

                # make QQ-plot for these columns
                (fig, ax) = plt.subplots()
                stats.probplot(transformed_data, dist="norm", plot=ax)
                ax.set_title(
                    f"QQ-plot for {column} (W: {round(new_stat, 3)}, p: {new_p}"
                )
                # plt.show()

    return data_to_trim


# Common blacklist genes that remain non-normal after transformation
blacklist = [
    "ERBB2",
    # "NKX2-1",
    # "RAD50",
    "ANXA1",
    "CASP8",
    "DIRAS3",
    "CAV1",
    "GATA3",
    "GATA6",
    "EIF4G1",
    "GATA6",
    "IGFBP2",
    "MS4A1",
    "RAB25",
    "BAK1",
    "BID",
    "BCL2L11",
    "ITGA2",
    "FASN",
    "G6PD",
    "IRS1",
    "XRCC5",
    "MYH11",
    "MRE11A",
    "RBM15",
    "TSC1",
    "KIT",
    "EEF2",
    "SQSTM1",
    "ERCC1",
]


def load_and_preprocess_data(rna_path, protein_path, cms_labels_path=None):
    """
    Load and preprocess omics data with optional CMS labeling

    Parameters:
    -----------
    rna_path : str
        Path to RNA data file
    protein_path : str
        Path to protein data file
    cms_labels_path : str, optional
        Path to CMS labels file. If None, no CMS labeling is performed

    Returns:
    --------
    tuple: Processed RNA and protein dataframes
    """
    # Load data
    rna_data = pd.read_csv(rna_path, sep="\t", index_col=0)
    protein_data = pd.read_csv(protein_path, sep="\t", index_col=0)

    # Transpose
    rna_data = rna_data.transpose()
    protein_data = protein_data.transpose()

    # Filter to keep only overlapping genes
    rna_data, protein_data = filter_dataframes(rna_data, protein_data)

    if cms_labels_path:
        # Apply CMS labels if provided
        classifier_labels = pd.read_csv(cms_labels_path, sep="\t")
        classifier_labels.rename(
            columns={classifier_labels.columns[0]: "sample_ID"}, inplace=True
        )
        classifier_labels = classifier_labels.loc[:, ["sample_ID", "SSP.nearestCMS"]]
        classifier_labels.set_index("sample_ID", inplace=True)

        rna_data = rna_data.join(classifier_labels)
        protein_data = protein_data.join(classifier_labels)

        # Create CMS subsets if labels exist
        rna_cms123 = rna_data.loc[
            rna_data.iloc[:, -1].isin(["CMS1", "CMS2", "CMS3"])
        ].drop(columns=["SSP.nearestCMS"])
        protein_cms123 = protein_data.loc[
            protein_data.iloc[:, -1].isin(["CMS1", "CMS2", "CMS3"])
        ].drop(columns=["SSP.nearestCMS"])

        return (
            rna_data.drop(columns=["SSP.nearestCMS"]),
            protein_data.drop(columns=["SSP.nearestCMS"]),
            rna_cms123,
            protein_cms123,
        )

    return rna_data, protein_data, None, None


def process_omics_data(rna_data, protein_data, output_prefix, blacklist=None):
    """
    Process omics data through normalization, transformation, and outlier handling

    Parameters:
    -----------
    rna_data : pd.DataFrame
        RNA expression data
    protein_data : pd.DataFrame
        Protein expression data
    output_prefix : str
        Prefix for output files
    blacklist : list, optional
        List of genes to exclude
    """
    # Center and scale
    rna_scaled = center_and_scale(rna_data)
    protein_scaled = center_and_scale(protein_data)

    # Transform and test for normality
    rna_transformed = transform_and_test(rna_scaled, "RNA_data")
    protein_transformed = transform_and_test(protein_scaled, "Protein_data")

    # Remove blacklisted genes if provided
    if blacklist:
        rna_transformed = rna_transformed.loc[
            :, ~rna_transformed.columns.isin(blacklist)
        ]
        protein_transformed = protein_transformed.loc[
            :, ~protein_transformed.columns.isin(blacklist)
        ]

    # Handle NaN values
    rna_transformed = rna_transformed.fillna(rna_transformed.mean())
    protein_transformed = protein_transformed.fillna(protein_transformed.mean())

    # Save processed data
    rna_transformed.to_csv(f"data/{output_prefix}_rna_processed.csv")
    protein_transformed.to_csv(f"data/{output_prefix}_protein_processed.csv")

    # Save gene list
    with open(f"data/{output_prefix}_gene_list.txt", "w") as f:
        for item in protein_transformed.columns:
            f.write(f"{item}\n")

    return rna_transformed, protein_transformed


def main():
    """
    Main function to process both CRC and glioma datasets
    """
    # # Example usage for CRC data
    # rna_crc, protein_crc, rna_cms123, protein_cms123 = load_and_preprocess_data(
    #     "data/LinkedOmics/linked_rna.cct",
    #     "data/LinkedOmics/linked_rppa.tsv",
    #     "data/LinkedOmics/TCGACRC_CMS_CLASSIFIER_LABELS.tsv",
    # )

    # # Process CRC data
    # blacklist = ["ERBB2", "NKX2-1", "RAD50"]
    # process_omics_data(rna_crc, protein_crc, "crc_cmsALL", blacklist)
    # if rna_cms123 is not None:
    #     process_omics_data(rna_cms123, protein_cms123, "crc_cms123", blacklist)

    # Example usage for glioma data
    rna_glioma, protein_glioma, _, _ = load_and_preprocess_data(
        "data/LinkedOmics/glioma_rna.cct", "data/LinkedOmics/glioma_rppa.cct"
    )

    # Process glioma data
    process_omics_data(rna_glioma, protein_glioma, "glioma")


if __name__ == "__main__":
    main()
