#Imports
from rdkit.Chem import PandasTools
from rdkit import Chem
import pandas as pd
import pickle
from pkasolver.chem import create_conjugate
from torch_geometric.data import Data
from pkasolver import constants as c
import torch
import numpy as np

#Functions
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
        pKa_type = df.marvin_pKa_type[i]
        pka = float(df.marvin_pKa[i])
        conjugates.append(create_conjugate(mol, index, pka))
    df["Conjugates"] = conjugates
    return df


def sort_conjugates(df):
    """Take DataFrame, check and correct the protonated and deprotonated molecules columns and return the new Dataframe."""
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

#################################################

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

##############################################

#Run Function

def preprocess(name, sdf):
    """Take name string and sdf path, process to Dataframe and save it as a pickle file."""
    df = import_sdf(sdf)
    df = conjugates_to_dataframe(df)
    df = sort_conjugates(df)
    return df
        
def preprocess_all(datasets, title='pd_all_datasets'):
    """Take dict of sdf paths, process to Dataframes and save it as a pickle file."""
    pd_datasets = {}
    for name, path in datasets.items(): 
        pd_datasets[name]=preprocess(name,path)
    return pd_datasets
    
        
        
#############################################


#define PairData Class
class PairData(Data):
    """Externsion of the Pytorch Geometric Data Class, which additionally takes a conjugated molecules in form of the edge_index2 and x2 input"""
    def __init__(self, edge_index, x, edge_index2, x2):
        super(PairData, self).__init__()
        self.edge_index = edge_index
        self.x = x
        self.edge_index2 = edge_index2
        self.x2 = x2

    def __inc__(self, key, value):
        if key == 'edge_index':
            return self.x.size(0)
        if key == 'edge_index2':
            return self.x2.size(0)
        else:
            return super().__inc__(key, value)

# Functions for converting Dataframe object to PyG Dataset
def make_nodes(mol, marvin_atom, n_features):
    """Take a rdkit.Mol, the atom index of the reaction center and a dict of node feature functions. 
    
    Return a torch.tensor with dimensions num_nodes(atoms) x num_node_features.
    """
    x = []
    i = 0
    for atom in mol.GetAtoms():
        node = []
        for feat in n_features.values():
            node.append(feat(atom, i, marvin_atom))
        x.append(node)
        i += 1
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
        edges.append(
            np.array(
                [
                    [bond.GetEndAtomIdx()],
                    [bond.GetBeginAtomIdx()],
                ]
            )
        )
        edge = []
        for feat in e_features.values():
            edge.append(feat(bond))
        edge_attr.extend([edge]*2)
        
    edge_index = torch.tensor(np.hstack(np.array(edges)), dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
    return edge_index, edge_attr

def mol_to_pairdata(row, n_features, e_features):
    """Take a DataFrame row, a dict of node feature functions and a dict of edge feature functions
    and return a Pytorch PairData object.
    """
    x_p = make_nodes(row.protonated, row.marvin_atom, n_features)
    edge_index_p, edge_attr_p = make_edges_and_attr(row.protonated, e_features)
    
    x_d = make_nodes(row.deprotonated, row.marvin_atom, n_features)
    edge_index_d, edge_attr_d = make_edges_and_attr(row.deprotonated, e_features)
    #return PairData(edge_index=edge_index_p, x=x_p, edge_index2=edge_index_d, x2=x_d, edge_attr=edge_attr_p, edge_attr2=edge_attr_d).to(device=device)
    data = PairData(edge_index_p, x_p, edge_index_d, x_d)
    data.edge_attr = edge_attr_p
    data.edge_attr2 = edge_attr_d
    return data

def mol_to_singledata(row, n_features, e_features):
    """Take a DataFrame row, a dict of node feature functions and a dict of edge feature functions
    and return a Pytorch Data object.
    """
    x = make_nodes(row.protonated, row.marvin_atom, n_features)
    edge_index, edge_attr = make_edges_and_attr(row.protonated, e_features)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def make_features_dicts(all_features, feat_list):
    """Take a dict of all features and a list of strings with all disered features
    and return a dict with these features
    """
    return {x:all_features[x] for x in feat_list}

def load_data(dataset, id_list):
    """Take a list of PyG Data objects and a list of molecule ids 
    and return a list of only the PyG Data objects that are in the list of ids. 
    """
    load_data =[]
    for data in dataset:
        if data.ID in id_list:
            load_data.append(data)
    return load_data

def make_pyg_dataset(df, list_n, list_e, paired=True):
    """Take a Dataframe, a list of strings of node features, a list of strings of edge features
    and return a List of PyG Data objects.
    
    Optional PairData by setting Paired=True.
    """
    n_feat = make_features_dicts(c.NODE_FEATURES, list_n)
    e_feat = make_features_dicts(c.EDGE_FEATURES, list_e)
    dataset = []
    if paired:
        func = mol_to_pairdata
    else:
        func = mol_to_singledata
    for i in range(len(df.index)):
        dataset.append(func(df.iloc[i], n_feat, e_feat))
        dataset[i].y = torch.tensor([float(df.pKa[i])], dtype=torch.float32)
        dataset[i].ID = df.ID[i]
    return dataset

def slice_list(input_list, size):
    input_size = len(input_list)
    slice_size = input_size // size
    remain = input_size % size
    result = []
    iterator = iter(input_list)
    for i in range(size):
        result.append([])
        for j in range(slice_size):
            result[i].append(next(iterator))
        if remain:
            result[i].append(next(iterator))
            remain -= 1
    return result

def cross_val_lists(sliced_lists, num):
    not_flattend = [x for i,x in enumerate(sliced_lists) if i!=num]
    train_list = [item for subl in not_flattend for item in subl]
    val_list = sliced_lists[num]
    return train_list, val_list




