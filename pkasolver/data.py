# Imports

from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import PandasTools, PropertyMol
from rdkit.Chem.AllChem import Compute2DCoords
from rdkit.Chem.PandasTools import LoadSDF

PandasTools.RenderImagesInAllDataFrames(images=True)
import random

import numpy as np
import pandas as pd
import torch
import tqdm
from torch_geometric.data import Data

from pkasolver.chem import create_conjugate
from pkasolver.constants import (
    DEVICE,
    EDGE_FEATURES,
    NODE_FEATURES,
    edge_feat_values,
    node_feat_values,
)


def load_data(base: str = "data/Baltruschat") -> dict:

    """Helper function loading the raw dataset"""

    sdf_filepath_training = f"{base}/combined_training_datasets_unique.sdf"
    sdf_filepath_novartis = f"{base}/novartis_cleaned_mono_unique_notraindata.sdf"
    sdf_filepath_Literture = f"{base}/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf"

    datasets = {
        "Training": sdf_filepath_training,
        "Novartis": sdf_filepath_novartis,
        "Literature": sdf_filepath_Literture,
    }
    return datasets


def train_validation_set_split(df: pd.DataFrame, ratio: float, seed=42):
    # splits a Dataframes rows randomly into two new Dataframes with a defined size ratio

    assert ratio > 0.0 and ratio < 1.0

    random.seed(seed)
    length = len(df)
    split = round(length * ratio)
    ids = list(range(length))
    random.shuffle(ids)
    train_ids = ids[:split]
    val_ids = ids[split:]
    train_df = df.iloc[train_ids, :]
    val_df = df.iloc[val_ids, :]
    return train_df, val_df


# data preprocessing functions - helpers
def import_sdf(sdf_filename: str):
    """Import an sdf file and return a Dataframe with an additional Smiles column."""
    df = LoadSDF(sdf_filename)
    for mol in df.ROMol:
        Compute2DCoords(mol)
    df["smiles"] = [Chem.MolToSmiles(m) for m in df["ROMol"]]
    return df


def conjugates_to_dataframe(df: pd.DataFrame):
    """Take DataFrame and return a DataFrame with a column of calculated conjugated molecules."""
    conjugates = []
    for i in tqdm.tqdm(range(len(df.index))):
        mol = df.ROMol[i]
        index = int(df.marvin_atom[i])
        pka = float(df.marvin_pKa[i])
        try:
            conj = create_conjugate(mol, index, pka, ignore_danger=True)
            conjugates.append(conj)
        except Exception as e:
            print(f"Could not create conjugate of mol number {i}")
            print(e)
            conjugates.append(mol)
    df["Conjugates"] = conjugates
    return df


def sort_conjugates(df):
    """Take DataFrame, check and correct the protonated and deprotonated molecules columns and return the new Dataframe."""
    prot = []
    deprot = []
    for i in range(len(df.index)):
        indx = int(
            df.marvin_atom[i]
        )  # mark reaction center where (de)protonation takes place
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
        else:
            print("prot = deprot")
            prot.append(mol)
            deprot.append(conj)
    df["protonated"] = prot
    df["deprotonated"] = deprot
    df = df.drop(columns=["ROMol", "Conjugates"])
    return df


# data preprocessing functions - main
def preprocess(sdf_filename: str):
    """Take name string and sdf path, process to Dataframe and save it as a pickle file."""
    df = import_sdf(sdf_filename)
    df = conjugates_to_dataframe(df)
    df = sort_conjugates(df)
    df["pKa"] = df["pKa"].astype(float)
    return df


def preprocess_all(sdf_files) -> dict:
    """Take dict of sdf paths, process to Dataframes and save it as a pickle file."""
    datasets = {}
    for name, sdf_filename in sdf_files.items():
        print(f"{name} : {sdf_filename}")
        print("###############")
        datasets[name] = preprocess(sdf_filename)
    return datasets


# Random Forrest/ML preparation functions
def make_stat_variables(df, X_list: list, y_name: list):
    """Take Pandas DataFrame and and return a Numpy Array of any other specified descriptors
    with size "Number of Molecules" x "Number of specified descriptors in X_list."
    """
    X = np.asfarray(df[X_list], float)
    y = np.asfarray(df[y_name], float).reshape(-1)
    return X, y


# Neural net data functions - helpers
class PairData(Data):
    """Externsion of the Pytorch Geometric Data Class, which additionally takes a conjugated molecules in form of the edge_index2 and x2 input"""

    def __init__(
        self,
        # NOTE: everything for protonated
        edge_index_p,
        edge_attr_p,
        x_p,
        charge_p,
        # everhtying for deprotonated
        edge_index_d,
        edge_attr_d,
        x_d,
        charge_d,
    ):
        super(PairData, self).__init__()
        self.edge_index_p = edge_index_p
        self.edge_index_d = edge_index_d

        self.x_p = x_p
        self.x_d = x_d

        self.edge_attr_p = edge_attr_p
        self.edge_attr_d = edge_attr_d

        self.charge_prot = charge_p
        self.charge_deprot = charge_d
        if x_p is not None:
            self.num_nodes = len(x_p)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_p":
            return self.x_p.size(0)
        if key == "edge_index_d":
            return self.x_d.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


from pandas.core.common import flatten


def calculate_nr_of_features(feature_list: list):
    i_n = 0
    if all(elem in node_feat_values for elem in feature_list):
        for feat in feature_list:
            i_n += len(node_feat_values[feat])
    elif all(elem in edge_feat_values for elem in feature_list):
        for feat in feature_list:
            i_n += len(edge_feat_values[feat])
    else:
        raise RuntimeError()
    return i_n


def make_nodes(mol, marvin_atom: int, n_features: dict):
    """Take a rdkit.Mol, the atom index of the reaction center and a dict of node feature functions.

    Return a torch.tensor with dimensions num_nodes(atoms) x num_node_features.
    """
    x = []
    for atom in mol.GetAtoms():
        node = []
        for feat in n_features.values():
            node.append(feat(atom, marvin_atom))
            # if atom.GetIdx() == int(marvin_atom):
            #    print(atom.GetSymbol())
        node = list(flatten(node))
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
        edges.append(np.array([[bond.GetBeginAtomIdx()], [bond.GetEndAtomIdx()],]))
        edges.append(np.array([[bond.GetEndAtomIdx()], [bond.GetBeginAtomIdx()],]))
        edge = []
        for feat in e_features.values():
            edge.append(feat(bond))
        edge = list(flatten(edge))
        edge_attr.extend([edge] * 2)

    edge_index = torch.tensor(np.hstack(np.array(edges)), dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
    return edge_index, edge_attr


def make_features_dicts(all_features, feat_list):
    """Take a dict of all features and a list of strings with all disered features
    and return a dict with these features
    """
    return {x: all_features[x] for x in feat_list}


# def mol_to_features_old(
#     row, n_features: dict, e_features: dict, protonation_state: str
# ):
#     if protonation_state == "protonated":
#         node = make_nodes(row.protonated, row.marvin_atom, n_features)
#         edge_index, edge_attr = make_edges_and_attr(row.protonated, e_features)
#         charge = np.sum([a.GetFormalCharge() for a in row.protonated.GetAtoms()])
#         return node, edge_index, edge_attr, charge
#     elif protonation_state == "deprotonated":
#         node = make_nodes(row.deprotonated, row.marvin_atom, n_features)
#         edge_index, edge_attr = make_edges_and_attr(row.deprotonated, e_features)
#         charge = np.sum([a.GetFormalCharge() for a in row.deprotonated.GetAtoms()])
#         return node, edge_index, edge_attr, charge
#     else:
#         raise RuntimeError()


def mol_to_features(mol, atom_idx: int, n_features: dict, e_features: dict):
    node = make_nodes(mol, atom_idx, n_features)
    edge_index, edge_attr = make_edges_and_attr(mol, e_features)
    charge = np.sum([a.GetFormalCharge() for a in mol.GetAtoms()])
    return node, edge_index, edge_attr, charge


def mol_to_paired_mol_data(
    prot: Chem.Mol, deprot: Chem.Mol, atom_idx: int, n_features: dict, e_features: dict,
):
    """Take a DataFrame row, a dict of node feature functions and a dict of edge feature functions
    and return a Pytorch PairData object.
    """
    node_p, edge_index_p, edge_attr_p, charge_p = mol_to_features(
        prot, atom_idx, n_features, e_features
    )
    node_d, edge_index_d, edge_attr_d, charge_d = mol_to_features(
        deprot, atom_idx, n_features, e_features
    )

    data = PairData(
        edge_index_p=edge_index_p,
        edge_attr_p=edge_attr_p,
        x_p=node_p,
        charge_p=charge_p,
        edge_index_d=edge_index_d,
        edge_attr_d=edge_attr_d,
        x_d=node_d,
        charge_d=charge_d,
    )
    return data


def mol_to_single_mol_data(
    mol, atom_idx: int, n_features: dict, e_features: dict,
):
    """Take a DataFrame row, a dict of node feature functions and a dict of edge feature functions
    and return a Pytorch Data object.
    """
    node_p, edge_index_p, edge_attr_p, charge = mol_to_features(
        mol, atom_idx, n_features, e_features
    )
    return Data(x=node_p, edge_index=edge_index_p, edge_attr=edge_attr_p), charge


def make_pyg_dataset_from_dataframe(
    df: pd.DataFrame, list_n: list, list_e: list, paired=False, mode: str = "all"
):
    """Take a Dataframe, a list of strings of node features, a list of strings of edge features
    and return a List of PyG Data objects.
    """
    print(f"Generating data with paired boolean set to: {paired}")

    if paired is False and mode not in ["protonated", "deprotonated"]:
        raise RuntimeError(f"Wrong combination of {mode} and {paired}")

    selected_node_features = make_features_dicts(NODE_FEATURES, list_n)
    selected_edge_features = make_features_dicts(EDGE_FEATURES, list_e)
    if paired:
        dataset = []
        for i in df.index:
            m = mol_to_paired_mol_data(
                df.protonated[i],
                df.deprotonated[i],
                df.marvin_atom[i],
                selected_node_features,
                selected_edge_features,
            )
            m.reference_value = torch.tensor([df.pKa[i]], dtype=torch.float32)
            m.ID = df.ID[i]
            m.to(device=DEVICE)  # NOTE: put everything on the GPU
            dataset.append(m)
        return dataset
    else:
        print(f"Generating data with {mode} form")
        dataset = []
        for i in df.index:
            if mode == "protonated":
                m, molecular_charge = mol_to_single_mol_data(
                    df.protonated[i],
                    df.marvin_atom[i],
                    selected_node_features,
                    selected_edge_features,
                )
            elif mode == "deprotonated":
                m, molecular_charge = mol_to_single_mol_data(
                    df.deprotonated[i],
                    df.marvin_atom[i],
                    selected_node_features,
                    selected_edge_features,
                )
            else:
                raise RuntimeError()
            m.reference_value = torch.tensor([df.pKa[i]], dtype=torch.float32)
            m.ID = df.ID[i]
            m.to(device=DEVICE)  # NOTE: put everything on the GPU
            dataset.append(m)
        return dataset


def make_paired_pyg_data_from_mol(
    mol: Chem.Mol, selected_node_features: dict, selected_edge_features: dict
):
    """Take a rdkit mol and generate a PyG Data object."""

    props = mol.GetPropsAsDict()
    try:
        pka = props["pKa"]
    except KeyError as e:
        print(f"No pKa found for molecule: {props}")
        print(props)
        print(Chem.MolToSmiles(mol))
        raise e
    if "epik_atom" in props.keys():
        atom_idx = props["epik_atom"]
    elif "marvin_atom" in props.keys():
        atom_idx = props["marvin_atom"]
    else:
        print(f"No reaction center foundfor molecule: {props}")
        print(props)
        print(Chem.MolToSmiles(mol))
        raise RuntimeError()

    try:
        conj = create_conjugate(mol, atom_idx, pka)
    except AssertionError as e:
        print(f"mol is failing because: {e}")
        raise e

    # sort mol and conj into protonated and deprotonated molecule
    if int(mol.GetAtomWithIdx(atom_idx).GetFormalCharge()) > int(
        conj.GetAtomWithIdx(atom_idx).GetFormalCharge()
    ):
        prot = mol
        deprot = conj
    else:
        prot = conj
        deprot = mol

    # create PairData object from prot and deprot with the selected node and edge features
    m = mol_to_paired_mol_data(
        prot, deprot, atom_idx, selected_node_features, selected_edge_features,
    )
    m.x = torch.tensor(pka, dtype=torch.float32)

    if "pka_number" in props.keys():
        m.pka_type = props["pka_number"]
    elif "marvin_pKa_type" in props.keys():
        m.pka_type = props["marvin_pKa_type"]
    else:
        m.pka_type = ""

    try:
        m.ID = props["ID"]
    except:
        m.ID = ""
    return m


def iterate_over_acids(
    acidic_mols_properties,
    nr_of_mols: int,
    partner_mol: Chem.Mol,
    nr_of_skipped_mols: int,
    pka_list: list,
    GLOBAL_COUNTER: int,
    pH: float,
    counter_list: list,
    smiles_list: list,
):

    acidic_mols = []
    skipping_acids = 0

    for idx, acid_prop in enumerate(
        reversed(acidic_mols_properties)
    ):  # list must be iterated in reverse, in order to protonated the strongest conjugate base first

        if skipping_acids == 0:  # if a acid was skipped, all further acids are skipped
            try:
                new_mol = create_conjugate(
                    partner_mol, acid_prop["atom_idx"], acid_prop["pka_value"], pH=pH,
                )
                Chem.SanitizeMol(new_mol)
                # new_mol = s.standardize(new_mol)

            except Exception as e:
                print(f"Error at molecule number {nr_of_mols} - acid enumeration")
                print(e)
                print(acid_prop)
                print(acidic_mols_properties)
                if partner_mol:
                    print(Chem.MolToSmiles(partner_mol))
                skipping_acids += 1
                nr_of_skipped_mols += 1
                continue  # continue instead of break, will not enter this routine gain since skipping_acids != 0

            pka_list.append(acid_prop["pka_value"])
            smiles_list.append(
                (Chem.MolToSmiles(new_mol), Chem.MolToSmiles(partner_mol))
            )

            for mol in [new_mol, partner_mol]:
                GLOBAL_COUNTER += 1
                counter_list.append(GLOBAL_COUNTER)
                mol.SetProp(f"CHEMBL_ID", str(acid_prop["chembl_id"]))
                mol.SetProp(f"INTERNAL_ID", str(GLOBAL_COUNTER))
                mol.SetProp(f"pKa", str(acid_prop["pka_value"]))
                mol.SetProp(f"epik_atom", str(acid_prop["atom_idx"]))
                mol.SetProp(f"pKa_number", f"acid_{idx + 1}")
                mol.SetProp(f"mol-smiles", f"{Chem.MolToSmiles(mol)}")

            # add current mol to list of acidic mol. for next
            # lower pKa value, this mol is starting structure
            acidic_mols.append(
                (
                    PropertyMol.PropertyMol(new_mol),
                    PropertyMol.PropertyMol(partner_mol),
                )
            )
            partner_mol = deepcopy(new_mol)

        else:
            skipping_acids += 1
    return acidic_mols, nr_of_skipped_mols, GLOBAL_COUNTER, skipping_acids


def iterate_over_bases(
    basic_mols_properties,
    nr_of_mols,
    partner_mol: Chem.Mol,
    nr_of_skipped_mols,
    pka_list: list,
    GLOBAL_COUNTER: int,
    pH: float,
    counter_list: list,
    smiles_list: list,
):

    basic_mols = []
    skipping_bases = 0
    for idx, basic_prop in enumerate(basic_mols_properties):
        if skipping_bases == 0:  # if a base was skipped, all further bases are skipped
            try:
                new_mol = create_conjugate(
                    partner_mol, basic_prop["atom_idx"], basic_prop["pka_value"], pH=pH,
                )

                Chem.SanitizeMol(new_mol)
                # new_mol = s.standardize(new_mol)

            except Exception as e:
                # in case error occurs new_mol is not in basic list
                print(f"Error at molecule number {nr_of_mols} - bases enumeration")
                print(e)
                print(basic_prop)
                print(basic_mols_properties)
                if partner_mol:
                    print(Chem.MolToSmiles(partner_mol))
                skipping_bases += 1
                nr_of_skipped_mols += 1
                continue

            pka_list.append(basic_prop["pka_value"])
            smiles_list.append(
                (Chem.MolToSmiles(partner_mol), Chem.MolToSmiles(new_mol))
            )

            for mol in [partner_mol, new_mol]:
                GLOBAL_COUNTER += 1
                counter_list.append(GLOBAL_COUNTER)
                mol.SetProp(f"CHEMBL_ID", str(basic_prop["chembl_id"]))
                mol.SetProp(f"INTERNAL_ID", str(GLOBAL_COUNTER))
                mol.SetProp(f"pKa", str(basic_prop["pka_value"]))
                mol.SetProp(f"epik_atom", str(basic_prop["atom_idx"]))
                mol.SetProp(f"pKa_number", f"acid_{idx + 1}")
                mol.SetProp(f"mol-smiles", f"{Chem.MolToSmiles(mol)}")

            # add current mol to list of acidic mol. for next
            # lower pKa value, this mol is starting structure
            basic_mols.append(
                (PropertyMol.PropertyMol(partner_mol), PropertyMol.PropertyMol(new_mol))
            )
            partner_mol = deepcopy(new_mol)

        else:
            skipping_bases += 1

    return basic_mols, nr_of_skipped_mols, GLOBAL_COUNTER, skipping_bases
