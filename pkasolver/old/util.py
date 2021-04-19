import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from pkasolver.analysis import compute_kl_divergence, compute_js_divergence


import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats

import torch
from torch_geometric.data import Data


def import_sdf(sdf):
    """Import an sdf file and return a Pandas Dataframe with an additional Smiles column."""

    df = PandasTools.LoadSDF(sdf)
    df["smiles"] = [Chem.MolToSmiles(m) for m in df["ROMol"]]
    return df


def morgan_fp(df, mol_column, neigh: int, nB: int, useFeatures=True):
    """Take Pandas DataFrame, creates Morgan fingerprints from the molecules in "mol_column"
    and add them to DataFrame.
    """
    df[mol_column + "_morganfp"] = [
        AllChem.GetMorganFingerprintAsBitVect(
            m, neigh, nBits=nB, useFeatures=useFeatures
        )
        for m in df[mol_column]
    ]
    return df


def make_fp_array(df, column_name):
    """Take Pandas DataFrame and return a Numpy Array
    with size "Number of Molecules" x "Bits of Morgan Fingerprint."
    """
    return np.array([np.array(row) for row in df[column_name]])


def make_stat_variables(df, X_list: list, y_name: list):
    """Take Pandas DataFrame and and return a Numpy Array of any other specified descriptors
    with size "Number of Molecules" x "Number of specified descriptors in X_list."
    """
    X = np.asfarray(df[X_list], float)
    y = np.asfarray(df[y_name], float).reshape(-1)
    return X, y


def cat_variables(X_feat, X_fp):
    """Take to Numpy Arrays and return an Array with the input Arrays concatinated along the columns."""
    return np.concatenate((X_feat, X_fp), axis=1)


def plot_results(prediction, true_vals, name: str):
    """Plot the prediction results in three subgraphs, showing the regression of y_hat against y."""

    a = {"y": true_vals, "y_hat": prediction}
    df = pd.DataFrame(data=a)

    r2 = (stats.pearsonr(df["y"], df["y_hat"])[0]) ** 2
    d = df["y"] - df["y_hat"]
    mse_f = np.mean(d ** 2)
    mae_f = np.mean(abs(d))
    rmse_f = np.sqrt(mse_f)
    r2_f = 1 - (sum(d ** 2) / sum((df["y"] - np.mean(df["y"])) ** 2))

    kl_div = compute_kl_divergence(df["y"], df["y_hat"], n_bins=20)
    js_div = compute_js_divergence(df["y"], df["y_hat"], n_bins=20)

    stat_info = f"""
    $r^2$ = {r2_f:.2} 
    MAE = {mae_f:.2}
    MASE = {mse_f:.2}
    RMSE = {rmse_f:.2}
    """

    dist_info = f"""
    kl divergence = {kl_div:.2}
    js divergence = {js_div:.2}
    """

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(name)
    plt.subplot(221)

    ax = sns.regplot(x="y", y="y_hat", data=df)
    ax.text(
        0,
        1,
        stat_info,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        linespacing=1,
    )
    ax.set_xlabel("$\mathrm{pKa}_{exp}$")
    ax.set_ylabel("$\mathrm{pKa}_{calc}$")

    plt.subplot(222)
    ax = sns.distplot(df["y"], bins=20, label="$\mathrm{pKa}_{exp}$")
    sns.distplot(df["y_hat"], bins=20, label="$\mathrm{pKa}_{calc}$")
    ax.set_xlabel("pKa")
    ax.text(
        0,
        1,
        dist_info,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        linespacing=1,
    )
    ax.legend()

    plt.subplot(223)
    ax = sns.distplot(df["y"] - df["y_hat"], bins=10)
    ax.set_xlabel("$\mathrm{pKa}_{exp} - \mathrm{pKa}_{calc}$")

    plt.show()
    plt.close()

def pka_to_ka(df):
    """Take DataFrame, calculate and add the "ka" to a new column and return the new Dataframe."""
    df["ka"] = [(10 ** -float(i)) for i in df.pKa]
    return df
