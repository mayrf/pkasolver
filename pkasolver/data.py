# Imports
from rdkit.Chem import PandasTools
from rdkit import Chem
import pandas as pd
from pkasolver.chemistry import create_conjugate
from torch_geometric.data import Data
from pkasolver.constants import NODE_FEATURES, EDGE_FEATURES
import torch
import numpy as np

# Functions
def import_sdf(sdf):
    """Import an sdf file and return a Dataframe with an additional Smiles column."""

    df = PandasTools.LoadSDF(sdf)
    df["smiles"] = [Chem.MolToSmiles(m) for m in df["ROMol"]]
    return df


def conjugates_to_dataframe(df: pd.DataFrame):
    """Take DataFrame and return a DataFrame with a column of calculated conjugated molecules."""
    conjugates = []
    for i in range(len(df.index)):
        mol = df.ROMol[i]
        index = int(df.marvin_atom[i])
        pka = float(df.marvin_pKa[i])
        conjugates.append(create_conjugate(mol, index, pka))
    df["Conjugates"] = conjugates
    return df


def sort_conjugates(df):
    """Take DataFrame, check and correct the protonated and deprotonated molecules columns and
    return the new Dataframe."""
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
        else:
            prot.append(mol)
            deprot.append(conj)
    df["protonated"] = prot
    df["deprotonated"] = deprot

    df = df.drop(columns=["ROMol", "Conjugates"])
    return df


# Run Function


def preprocess(file_path: str) -> pd.DataFrame:
    """Take filepath to sdf file, returns processed Dataframe"""
    df = import_sdf(file_path)
    df = conjugates_to_dataframe(df)
    df = sort_conjugates(df)
    return df


def preprocess_all(datasets) -> dict:
    """Take dict of sdf paths, process to Dataframes and save it as a pickle file."""
    dic = {}
    for name, path in datasets.items():
        df = preprocess(path)
        dic[name] = df
    return dic


#############################################


class PairData(Data):
    """not really, but essentially a Dataclass"""

    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2


# Functions for converting Dataframe object to PyG Dataset
def make_nodes(mol, marvin_atom, n_features):
    """Take a rdkit.Mol, the atom index of the reaction center and a dict of node feature functions.

    Return a torch.tensor with dimensions num_nodes(atoms) x num_node_features.
    """
    x = []
    for i, atom in enumerate(mol.GetAtoms()):
        node = []
        for feat in n_features.values():
            node.append(feat(atom, i, marvin_atom))
        x.append(node)
    return torch.tensor(np.array([np.array(xi) for xi in x]), dtype=torch.float)


def make_edges_and_attr(mol, e_features):
    """Take a rdkit.Mol and a dict of edge feature functions.

    Return a torch.tensor with dimensions 2 x num_edges
    and a torch.tensor with dimensions num_edges x num_edge_features.
    """
    edges = []
    edge_attr = []
    for bond in mol.GetBonds():
        edges.append(
            np.array(
                [
                    [bond.GetBeginAtomIdx()],
                    [bond.GetEndAtomIdx()],
                ]
            )
        )
        edge = []
        for feat in e_features.values():
            edge.append(feat(bond))
        edge_attr.append(edge)

    edge_index = torch.tensor(np.hstack(np.array(edges)), dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
    return edge_index, edge_attr


# def mol_to_pairdata(row, n_features, e_features):
#     """Take a DataFrame row, a dict of node feature functions and a dict of edge feature functions
#     and return a Pytorch PairData object.
#     """
#     x_p = make_nodes(row.protonated, row.marvin_atom, n_features)
#     edge_index_p, edge_attr_p = make_edges_and_attr(row.protonated, e_features)

#     x_d = make_nodes(row.deprotonated, row.marvin_atom, n_features)
#     edge_index_d, edge_attr_d = make_edges_and_attr(row.deprotonated, e_features)
#     # return PairData(edge_index=edge_index_p, x=x_p, edge_index2=edge_index_d, x2=x_d, edge_attr=edge_attr_p, edge_attr2=edge_attr_d).to(device=device)
#     data = PairData(edge_index_p, x_p, edge_index_d, x_d)
#     data.edge_attr = edge_attr_p
#     data.edge_attr2 = edge_attr_d
#     return data


def mol_to_data(row, n_features, e_features, protonation_state: str = "protonated"):
    """Take a DataFrame row, a dict of node feature functions and a dict of edge feature functions
    and return a Pytorch Data object.
    """
    assert protonation_state in ("protonated", "deprotonated")
    if protonation_state == "protonated":
        x = make_nodes(row.protonated, row.marvin_atom, n_features)
        edge_index, edge_attr = make_edges_and_attr(row.protonated, e_features)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.charge = np.sum([a.GetFormalCharge() for a in row.protonated.GetAtoms()])
        return data
    else:
        x = make_nodes(row.deprotonated, row.marvin_atom, n_features)
        edge_index, edge_attr = make_edges_and_attr(row.deprotonated, e_features)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.charge = np.sum([a.GetFormalCharge() for a in row.protonated.GetAtoms()])
        return data


def make_features_dicts(all_features, feat_list):
    """Take a dict of all features and a list of strings with all disered features
    and return a dict with these features
    """
    return {x: all_features[x] for x in feat_list}


def load_data(dataset, id_list):
    """Take a list of PyG Data objects and a list of molecule ids
    and return a list of only the PyG Data objects that are in the list of ids.
    """
    load_data = []
    for data in dataset:
        if data.ID in id_list:
            load_data.append(data)
    return load_data


def make_pyg_dataset(df, node_features, edge_features, paired=False):
    """Take a Dataframe, a list of strings of node features, a list of strings of edge features
    and return a List of PyG Data objects.

    Optional PairData by setting Paired=True.
    """
    selected_node_features = make_features_dicts(NODE_FEATURES, node_features)
    selected_edge_features = make_features_dicts(EDGE_FEATURES, edge_features)
    dataset = []
    for i in range(len(df.index)):
        m_prot = mol_to_data(
            df.iloc[i],
            selected_node_features,
            selected_edge_features,
            protonation_state="protonated",
        )
        m_deprot = mol_to_data(
            df.iloc[i],
            selected_node_features,
            selected_edge_features,
            protonation_state="deprotonated",
        )

        dataset.append(PairData(m_prot, m_deprot))
        dataset[i].y = torch.tensor([float(df.pKa[i])], dtype=torch.float32)
        dataset[i].ID = df.ID[i]
    return dataset
