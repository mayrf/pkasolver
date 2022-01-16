# Imports

from copy import deepcopy
from typing import Tuple

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
    """Helper function that takes path to working directory and
    returns a dictionary containing the paths to the training and testsets.

    Parameters
    ----------
    base
        path to folder containing dataset sd files

    Returns
    ----------
    dict
        keys (str): dataset names
        values (str): dataset abs. paths

    """
    sdf_filepath_training = f"{base}/combined_training_datasets_unique.sdf"
    sdf_filepath_novartis = f"{base}/novartis_cleaned_mono_unique_notraindata.sdf"
    sdf_filepath_Literture = f"{base}/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf"

    datasets = {
        "Training": sdf_filepath_training,
        "Novartis": sdf_filepath_novartis,
        "Literature": sdf_filepath_Literture,
    }
    return datasets


# data preprocessing functions - helpers
def import_sdf(sd_filename: str) -> pd.DataFrame:
    """Imports an sd file and returns a Dataframe with an additional Smiles column.

    Parameters
    ----------
    sd_filename
        dataset path

    Returns
    ----------
    pd.DataFrame
        DataFrame of dataset with Chem.rdchem.Mol object and molecule properties as columns

    """

    df = LoadSDF(sd_filename)
    for mol in df.ROMol:
        Compute2DCoords(mol)
    df["smiles"] = [Chem.MolToSmiles(m) for m in df["ROMol"]]
    return df


def conjugates_to_dataframe(
    df: pd.DataFrame, mol_col: str = "ROMol", ph: float = 7.4
) -> pd.DataFrame:
    """Takes DataFrame and returns a DataFrame with a column of calculated conjugated molecules.
    Molecule properties must contain columns "marvin_atom" and "marvin_pka".

    Parameters
    ----------
    df
        DataFrame object
    mol_col
        name of column containing the Chem.rdchem.Mol objects
    ph
        ph of the protonation state of the Chem.rdchem.Mol objects

    Returns
    -------
    pd.DataFrame
        df with additional "Conjugates" column

    """
    conjugates = []
    for i in tqdm.tqdm(range(len(df.index))):
        mol = df[mol_col][i]
        index = int(df.marvin_atom[i])
        pka = float(df.marvin_pKa[i])
        try:
            conj = create_conjugate(mol, index, pka, ignore_danger=True, pH=ph)
            conjugates.append(conj)
        except Exception as e:
            print(f"Could not create conjugate of mol number {i}")
            print(e)
            conjugates.append(mol)
    df["Conjugates"] = conjugates
    return df


def sort_conjugates(
    df: pd.DataFrame, mol_col_1: str = "ROMol", mol_col_2: str = "Conjugates"
) -> pd.DataFrame:
    """Takes DataFrame, sorts the molecules in the two specified columns into
    two new columns, "protonated" and "deprotonated" that replace the two old columns.
    Molecule properties must contain columns "marvin_atom".

    Parameters
    ----------
    df
        DataFrame object
    mol_col_1, mol_col_2
        name of the columns containing the Chem.rdchem.Mol objects

    Returns
    -------
    pd.DataFrame
        df with the two columns specified, replaced by the columns "protonated" and "deprotonated"

    """
    prot = []
    deprot = []
    for i in range(len(df.index)):
        indx = int(
            df.marvin_atom[i]
        )  # mark reaction center where (de)protonation takes place
        mol = df[mol_col_1][i]
        conj = df[mol_col_2][i]
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
def preprocess(sd_filename: str, ph=7.4) -> pd.DataFrame:
    """Takes path of sdf file containing pkadata and returns a dataframe with column for protonated and deprotonated molecules.
    Molecule properties must contain columns "marvin_atom" and "marvin_pka".

    Parameters
    ----------
    sd_filename
        dataset path
    ph
        ph of the protonation state of the Chem.rdchem.Mol objects

    Returns
    -------
    pd.DataFrame
        DataFrame with molecule properties and "protonated" and "deprotonated" Chem.rdchem.Mol objects as columns
    """
    df = import_sdf(sd_filename)
    df = conjugates_to_dataframe(df, ph=ph)
    df = sort_conjugates(df)
    df["pKa"] = df["pKa"].astype(float)
    return df


def preprocess_all(sd_files: dict, ph=7.4) -> dict:
    """Takes dictionary of pka data sets containing paths to sdf files, preprocesses them to Dataframes
    with protonated and deprotonated molecules and returns them in a dictionary.

    Parameters
    ----------
    sd_files
        keys (str): dataset name
        values (str): dataset path

    ph
        ph of the protonation state of the Chem.rdchem.Mol objects

    Returns
    -------
    dict
        keys (str): dataset name
        values (pd.DataFrame):  DataFrames with molecule properties and "protonated"
                                and "deprotonated" Chem.rdchem.Mol objects as columns

    """
    datasets = {}
    for name, sd_filename in sd_files.items():
        print(f"{name} : {sd_filename}")
        print("###############")
        datasets[name] = preprocess(sd_filename, ph=ph)
    return datasets


# Neural net data functions - helpers
class PairData(Data):
    """Extension of the Pytorch Geometric Data Class, which additionally
    takes a conjugated molecules.

    Attributes
    -------
    edge_index_p (torch.Tensor)

    x_p, x_d (torch.Tensor)
        Node feature matrix with shape [num_nodes, num_node_features]
    edge_index_p, edge_index_d (torch.Tensor)
        Graph connectivity in COO format with shape [2, num_edges]
    edge_attr_p, edge_attr_d (torch.Tensor)
        Edge feature matrix with shape [num_edges, num_edge_features]
    charge_p, charge_d (int)
        molecular charge

    """

    def __init__(
        self,
        # NOTE: everything for protonated
        edge_index_p,
        edge_attr_p,
        x_p,
        charge_p,
        # everything for deprotonated
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


def calculate_nr_of_features(feature_list: list) -> int:
    """Calculates number of nodes and edge one hot features from input list.

    Parameters
    ----------
    features_list
        list of features (must all be contained either in global dict node_feat_values or edge_feat_values)

    Returns
    -------
    int
        number of feature values

    """
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


def make_nodes(mol: Chem.rdchem.Mol, atom_idx: int, n_features: dict) -> torch.Tensor:
    """Takes an Chem.rdchem.Mol object, the atom index of the reaction center and a dictionary of node feature functions
    and returns a torch.tensor of node features for all atoms of one molecule.

    Parameters
    ----------
    mol
        input molecule
    atom_idx
        atom index of ionization center
    n_features
        dictionary containing functions for node feature generation

    Returns
    -------
    torch.Tensor
        tensor with dimensions num_nodes(atoms) x num_node_features.

    """
    x = []
    for atom in mol.GetAtoms():
        node = []
        for feat in n_features.values():
            node.append(feat(atom, atom_idx))
        node = list(flatten(node))
        x.append(node)
    return torch.tensor(np.array([np.array(xi) for xi in x]), dtype=torch.float)


def make_edges_and_attr(
    mol: Chem.rdchem.Mol, e_features
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Takes an Chem.rdchem.Mol and a dictionary of edge feature functions
    and returns one tensor containing adjacency information and one of edge features for all edges of one molecule.

    Parameters
    ----------
    mol
        input molecule
    e_features
        dictionary containing functions for edge feature generation

    Returns
    -------
    edge_index
        tensor of dimension 2 x num_edges
    edge_attr
        tensor with dimensions num_edge) x num_edge_features.

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
        edge = list(flatten(edge))
        edge_attr.extend([edge] * 2)

    edge_index = torch.tensor(np.hstack(np.array(edges)), dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
    return edge_index, edge_attr


def make_features_dicts(all_features: dict, feat_list: list) -> dict:
    """Takes a dict of all features and a list of strings with all desired features
    and returns a dict with these features.

    ----------
    all_features
        dictionary of all available features and their possible values
    feat_list
        list of feature names to be filtered for

    Returns
    -------
    dict
        filtered features dictionary

    """
    return {x: all_features[x] for x in feat_list}


def mol_to_features(
    mol: Chem.rdchem.Mol, atom_idx: int, n_features: dict, e_features: dict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Creates the node and edge feature tensors from the input molecule.

    Parameters
    ----------
    mol
        input molecule
    atom_idx
        atom index of ionization center
    n_features
        dictionary containing functions for node feature generation
    e_features
        dictionary containing functions for node feature generation

    Returns
    -------
    nodes
        tensor with dimensions num_nodes(atoms) x num_node_features.
    edge_index
        tensor of dimension 2 x num_edges
    edge_attr
        tensor with dimensions num_edge) x num_edge_features.
    charge
        molecule charge

    """
    nodes = make_nodes(mol, atom_idx, n_features)
    edge_index, edge_attr = make_edges_and_attr(mol, e_features)
    charge = np.sum([a.GetFormalCharge() for a in mol.GetAtoms()])
    return nodes, edge_index, edge_attr, charge


def mol_to_paired_mol_data(
    prot: Chem.rdchem.Mol,
    deprot: Chem.rdchem.Mol,
    atom_idx: int,
    n_features: dict,
    e_features: dict,
) -> PairData:
    """Take a DataFrame row, a dict of node feature functions and a dict of edge feature functions
    and return a Pytorch PairData object.

    Parameters
    ----------
    prot
        protonated rdkit mol object
    deprot
        deprotonated rdkit mol object
    atom_idx
        ionization center atom index
    n_features
        dictionary of node features
    e_features
        dictionary of edge features

    Returns
    -------
    PairData
        Data object ready for use with Pytorch Geometric models

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
    mol: Chem.rdchem.Mol,
    atom_idx: int,
    n_features: dict,
    e_features: dict,
):
    """Take a DataFrame row, a dict of node feature functions and a dict of edge feature functions
    and return a Pytorch Data object.

    Parameters
    ----------
    mol
        rdkit mol object
    atom_idx
        ionization center atom index
    n_features
        dictionary of node features
    e_features
        dictionary of edge features

    Returns
    -------
    PairData
        Data object ready for use with Pytorch Geometric models

    """
    node_p, edge_index_p, edge_attr_p, charge = mol_to_features(
        mol, atom_idx, n_features, e_features
    )
    return Data(x=node_p, edge_index=edge_index_p, edge_attr=edge_attr_p), charge


def make_pyg_dataset_from_dataframe(
    df: pd.DataFrame, list_n: list, list_e: list, paired=False, mode: str = "all"
) -> list:
    """Take a Dataframe, a list of strings of node features, a list of strings of edge features
    and return a List of PyG Data objects.

    Parameters
    ----------
    df
        DataFrame containing "protonated" and "deprotonated" columns with mol objects, as well as "pKa" and "marvin_atom" columns
    list_n
        list of node features to be used
    list_e
        list of edge features to be used
    paired
        If true, including protonated and deprotonated molecules, if False only the type specified in mode
    mode
        if paired id false, use data from columnname == mol

    Returns
    -------
    list
        contains all molecules from df as pyG Graph data

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
) -> PairData:
    """Generate a PyG Data object from an input molecule and feature dictionaries.

    Parameters
    ----------
    mol
        input molecule
    selected_node_features
        dictionary of selected node features
    selected_edge_features
        dictionary of selected edge features

    Returns
    -------
    PairData
        Data object ready for use with Pytorch Geometric models

    """
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
        prot,
        deprot,
        atom_idx,
        selected_node_features,
        selected_edge_features,
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
    acidic_mols_properties: list,
    nr_of_mols: int,
    partner_mol: Chem.Mol,
    nr_of_skipped_mols: int,
    pka_list: list,
    GLOBAL_COUNTER: int,
    pH: float,
    counter_list: list,
    smiles_list: list,
) -> Tuple[list, int, int, int]:
    """Processes the acidic pKa values of an Schrödinger EPIK pka query
    and returns a pair of protonated and deprotonated molecules for every
    pKa. Takes and updates global counters and skip trackers.

    Parameters
    ----------
    acidic_mols_properties
        list of of dictionaries, each containing pka, ionization index
        and CHEMBL id of an acidic input molecule
    nr_of_mols
        index number of molecule
    partner_mol
        molecule in protonation state at pH=pH
    nr_of_skipped_mols
        global number of skipped molecules
    pka_list
        list of all pKas already found for this molecule
    GLOBAL_COUNTER
        counts total number of pKas processed
    pH
        pH of protonations state of partner_mol
    counter_list: list,
        list of values of GLOBAL_COUNTER
    smiles_list: list,
        list of smiles tuples of protonated and deprotonated molecules of all pka reactions

    Returns
    -------
    acidic_mols (list)
        list of tuples of protonated and deprotonated molecules for
        every pka in acidic_mols_properties that did not yield an error
    nr_of_skipped_mols (int)
        global number of skipped molecules
    GLOBAL_COUNTER (int)
        counts total number of pKas processed
    skipping_acids
        number of acids skipped (max 1)

    """
    acidic_mols = []
    skipping_acids = 0

    for idx, acid_prop in enumerate(
        reversed(acidic_mols_properties)
    ):  # list must be iterated in reverse, in order to protonated the strongest conjugate base first

        if skipping_acids == 0:  # if a acid was skipped, all further acids are skipped
            try:
                new_mol = create_conjugate(
                    partner_mol,
                    acid_prop["atom_idx"],
                    acid_prop["pka_value"],
                    pH=pH,
                )
                Chem.SanitizeMol(new_mol)

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
    basic_mols_properties: Chem.Mol,
    nr_of_mols: int,
    partner_mol: Chem.Mol,
    nr_of_skipped_mols,
    pka_list: list,
    GLOBAL_COUNTER: int,
    pH: float,
    counter_list: list,
    smiles_list: list,
) -> Tuple[list, int, int, int]:
    """Processes the basic pKa values of an Schrödinger EPIK pka query
    and returns a pair of protonated and deprotonated molecules for every
    pKa. Takes and updates global counters and skip trackers.

    Parameters
    ----------
    basic_mols_properties
        list of of dictionaries, each containing pka, ionization index
        and CHEMBL id of an basic input molecule
    nr_of_mols
        index number of molecule
    partner_mol
        molecule in protonation state at pH=pH
    nr_of_skipped_mols
        global number of skipped molecules
    pka_list
        list of all pKas already found for this molecule
    GLOBAL_COUNTER
        counts total number of pKas processed
    pH
        pH of protonations state of partner_mol
    counter_list: list,
        list of values of GLOBAL_COUNTER
    smiles_list: list,
        list of smiles tuples of protonated and deprotonated molecules of all pka reactions

    Returns
    -------
    basic_mols (list)
        list of tuples of protonated and deprotonated molecules for
        every pka in basic_mols_properties that did not yield an error
    nr_of_skipped_mols (int)
        global number of skipped molecules
    GLOBAL_COUNTER (int)
        counts total number of pKas processed
    skipping_bases
        number of bases skipped (max 1)

    """
    basic_mols = []
    skipping_bases = 0
    for idx, basic_prop in enumerate(basic_mols_properties):
        if skipping_bases == 0:  # if a base was skipped, all further bases are skipped
            try:
                new_mol = create_conjugate(
                    partner_mol,
                    basic_prop["atom_idx"],
                    basic_prop["pka_value"],
                    pH=pH,
                )

                Chem.SanitizeMol(new_mol)

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
