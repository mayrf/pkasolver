import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem
from pkasolver.analysis import compute_kl_divergence, compute_js_divergence


import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats

import torch
from torch_geometric.data import Data


def import_sdf(sdf):
    """
    Import sdf file into a pandas Dataframe
    """

    df = PandasTools.LoadSDF(sdf)
    df["smiles"] = [Chem.MolToSmiles(m) for m in df["ROMol"]]
    return df


def morgan_fp(df, mol_column, neigh: int, nB: int, useFeatures=True):
    """creates morgan fingerprints and adds them to DataFrame"""
    df[mol_column + "_morganfp"] = [
        AllChem.GetMorganFingerprintAsBitVect(
            m, neigh, nBits=nB, useFeatures=useFeatures
        )
        for m in df[mol_column]
    ]
    return df


def make_fp_array(df, column_name):
    """Creats a numpy array of the morgan fingerprint with each bit in a separate column"""
    return np.array([np.array(row) for row in df[column_name]])


def make_stat_variables(df, X_list: list, y_name: list):
    """Creats a numpy array of any other specified descriptors"""
    X = np.asfarray(df[X_list], float)
    y = np.asfarray(df[y_name], float).reshape(-1)
    return X, y


def cat_variables(X_feat, X_fp):
    """Concatinates an array of descriptors with an array of morgan fingerprints"""
    return np.concatenate((X_feat, X_fp), axis=1)


def plot_results(prediction, true_vals, name: str):
    """Plots the prediction results in 3 sub graphs showing the regression of y_hat against y """
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


def create_conjugate(mol, id, pka, pH=7.4):
    """create a new molecule that is the conjugated base/acid to the input molecule"""
    mol = Chem.RWMol(mol)
    atom = mol.GetAtomWithIdx(id)
    charge = atom.GetFormalCharge()
    Ex_Hs = atom.GetNumExplicitHs()
    Tot_Hs = atom.GetTotalNumHs()

    #
    if pka > pH and Tot_Hs > 0:
        atom.SetFormalCharge(charge - 1)
        if Ex_Hs > 0:
            atom.SetNumExplicitHs(Ex_Hs - 1)
    elif pka < pH:
        atom.SetFormalCharge(charge + 1)
        if Tot_Hs == 0 or Ex_Hs > 0:
            atom.SetNumExplicitHs(Ex_Hs + 1)
    else:
        # pka > pH and Tot_Hs < 0
        atom.SetFormalCharge(charge + 1)
        if Tot_Hs == 0 or Ex_Hs > 0:
            atom.SetNumExplicitHs(Ex_Hs + 1)

    atom.UpdatePropertyCache()
    return mol


def conjugates_to_DataFrame(df: pd.DataFrame):
    """
     [summary]

    Returns
    -------
    [type]
        [description]
    """

    #  |   |    |    |    | -> what's in
    #
    #

    conjugates = []
    for i in range(len(df.index)):
        mol = df.ROMol[i]
        index = int(df.marvin_atom[i])
        pKa_type = df.marvin_pKa_type[i]
        pka = float(df.marvin_pKa[i])
        conjugates.append(create_conjugate(mol, index, pka))
    df["Conjugates"] = conjugates
    return df


def sort_conjugates(df):
    """sorts the input DataFrame so that the protonated and deprotonated molecules are in their corresponding columns"""

    prot = []
    deprot = []
    for i in range(len(df.index)):
        indx = int(df.marvin_atom[i])
        mol = df.ROMol[i]
        conj = df.Conjugates[i]

        charge_mol = int(mol.GetAtomWithIdx(indx).GetFormalCharge())
        charge_conj = int(conj.GetAtomWithIdx(indx).GetFormalCharge())

        if charge_mol < charge_conj:
            prot.append(conj)
            deprot.append(mol)
        elif charge_mol > charge_conj:
            prot.append(mol)
            deprot.append(conj)
    df["protonated"] = prot
    df["deprotonated"] = deprot
    df = df.drop(columns=["ROMol", "Conjugates"])
    return df


def pka_to_ka(df):
    df["ka"] = [(10 ** -float(i)) for i in df.pKa]
    return df


def mol_to_pyg(prot, deprot):

    i = 0
    num_atoms = prot.GetNumAtoms()
    nodes = []
    edges = []
    edges_attr = []

    for mol in [prot, deprot]:

        for atom in mol.GetAtoms():
            nodes.append(
                list(
                    (
                        atom.GetIdx() + num_atoms * i,
                        atom.GetAtomicNum(),
                        atom.GetFormalCharge(),
                        atom.GetChiralTag(),
                        atom.GetHybridization(),
                        atom.GetNumExplicitHs(),
                        atom.GetIsAromatic(),
                    )
                )
            )

        for bond in mol.GetBonds():
            edges.append(
                np.array(
                    [
                        [bond.GetBeginAtomIdx() + num_atoms * i],
                        [bond.GetEndAtomIdx() + num_atoms * i],
                    ]
                )
            )
            edges.append(
                np.array(
                    [
                        [bond.GetEndAtomIdx() + num_atoms * i],
                        [bond.GetBeginAtomIdx() + num_atoms * i],
                    ]
                )
            )
            bond_type = bond.GetBondType()
            edges_attr.append(bond_type)
            edges_attr.append(bond_type)

        i += 1

    X = torch.tensor(np.array([np.array(xi) for xi in nodes]), dtype=torch.float)
    edge_index = torch.tensor(np.hstack(np.array(edges)), dtype=torch.long)
    edge_attr = torch.tensor(np.array(edges_attr).T, dtype=torch.float)

    return Data(x=X, edge_index=edge_index, edge_attr=edge_attr)
