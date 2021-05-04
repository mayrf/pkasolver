import pkasolver
import pytest
import sys
import numpy as np
import pandas as pd


def load_data():

    """Test loading the raw dataset"""

    base = "data/baltruschat"
    sdf_training = f"{base}/combined_training_datasets_unique.sdf"
    sdf_novartis = f"{base}/novartis_cleaned_mono_unique_notraindata.sdf"
    sdf_AvLiLuMoVe = f"{base}/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf"

    datasets = {
        "Training": sdf_training,
        "Novartis": sdf_novartis,
        "AvLiLuMoVe": sdf_AvLiLuMoVe,
    }
    return datasets

def test_data_processing():
    """Test the basic data processing. Reading in the sdf files, generating
    all properties that are needed and making some basic checks to test
    that things are done correctly and reproducibly."""

    from ..data import import_sdf, conjugates_to_dataframe, sort_conjugates
    from rdkit import Chem

    base = "data/baltruschat"
    sdf_training = f"{base}/combined_training_datasets_unique.sdf"
    df = import_sdf(sdf_training)
    # test against the frist two pka values
    assert np.isclose(float(df.iloc[0]["pKa"]), 6.21, atol=1e-9)
    assert np.isclose(float(df.iloc[1]["pKa"]), 7.46, atol=1e-9)

    # test that conjugates are added
    df = conjugates_to_dataframe(df)
    assert "Conjugates" in df.columns
    df = sort_conjugates(df)
    assert df.size == 53946
    assert df.shape == (5994, 9)
    assert df.ndim == 2
    # test that conjugates works for a few examples
    # Example 1
    m_prot = df.iloc[0]["protonated"]
    m_deprot = df.iloc[0]["deprotonated"]
    m_prot_smiles = Chem.MolToSmiles(m_prot)
    m_deprot_smiles = Chem.MolToSmiles(m_deprot)
    assert m_prot_smiles == "Brc1c(N2CCCCCC2)nc(C2CC2)[nH+]c1NC1CC1"
    assert m_deprot_smiles == "Brc1c(NC2CC2)nc(C2CC2)nc1N1CCCCCC1"
    m_prot_smiles_charge = np.sum([a.GetFormalCharge() for a in m_prot.GetAtoms()])
    m_deprot_smiles_charge = np.sum([a.GetFormalCharge() for a in m_deprot.GetAtoms()])
    assert m_prot_smiles_charge == 1
    assert m_deprot_smiles_charge == 0
    # Example 2
    m_prot = df.iloc[11]["protonated"]
    m_deprot = df.iloc[11]["deprotonated"]
    m_prot_smiles = Chem.MolToSmiles(m_prot)
    m_deprot_smiles = Chem.MolToSmiles(m_deprot)
    assert m_prot_smiles == "Brc1ccc(Nc2c3ccccc3[nH+]c3ccccc23)cc1"
    assert m_deprot_smiles == "Brc1ccc(Nc2c3ccccc3nc3ccccc23)cc1"
    m_prot_smiles_charge = np.sum([a.GetFormalCharge() for a in m_prot.GetAtoms()])
    m_deprot_smiles_charge = np.sum([a.GetFormalCharge() for a in m_deprot.GetAtoms()])
    assert m_prot_smiles_charge == 1
    assert m_deprot_smiles_charge == 0
    # Example 3
    m_prot = df.iloc[1234]["protonated"]
    m_deprot = df.iloc[1234]["deprotonated"]
    m_prot_smiles = Chem.MolToSmiles(m_prot)
    m_deprot_smiles = Chem.MolToSmiles(m_deprot)
    assert m_prot_smiles == "CCCCCCCCCCCCCCC[NH3+]"
    assert m_deprot_smiles == "CCCCCCCCCCCCCCCN"
    m_prot_smiles_charge = np.sum([a.GetFormalCharge() for a in m_prot.GetAtoms()])
    m_deprot_smiles_charge = np.sum([a.GetFormalCharge() for a in m_deprot.GetAtoms()])
    assert m_prot_smiles_charge == 1
    assert m_deprot_smiles_charge == 0
    # Example 4
    m_prot = df.iloc[2000]["protonated"]
    m_deprot = df.iloc[2000]["deprotonated"]
    m_prot_smiles = Chem.MolToSmiles(m_prot)
    m_deprot_smiles = Chem.MolToSmiles(m_deprot)
    assert (
        m_prot_smiles
        == "CN(C)C(=O)Cn1c(-c2ccccc2)c(C2CCCCC2)c2ccc(-c3noc(=O)[nH]3)cc21"
    )
    assert (
        m_deprot_smiles
        == "CN(C)C(=O)Cn1c(-c2ccccc2)c(C2CCCCC2)c2ccc(-c3noc(=O)[n-]3)cc21"
    )
    m_prot_smiles_charge = np.sum([a.GetFormalCharge() for a in m_prot.GetAtoms()])
    m_deprot_smiles_charge = np.sum([a.GetFormalCharge() for a in m_deprot.GetAtoms()])
    assert m_prot_smiles_charge == 0
    assert m_deprot_smiles_charge == -1


def test_load_data():
    load_data()


def test_make_feature_list():
    from ..data import make_features_dicts
    from ..constants import NODE_FEATURES, EDGE_FEATURES

    node_features = [
        "atomic_number",
        "hybridization",
    ]
    edge_features = ["bond_type", "is_conjugated"]

    f = make_features_dicts(NODE_FEATURES, node_features)
    assert "atomic_number" in f.keys()
    assert "hybridization" in f.keys()
    f = make_features_dicts(EDGE_FEATURES, edge_features)
    assert "bond_type" in f.keys()
    assert "is_conjugated" in f.keys()


def generate_pairwise_data():
    from ..data import make_pyg_dataset, import_sdf
    from ..data import import_sdf, conjugates_to_dataframe, sort_conjugates

    node_features = [
        "atomic_number",
        "hybridization",
    ]
    edge_features = ["bond_type", "is_conjugated"]
    dataset = load_data()
    df_training = import_sdf(dataset["Training"])
    df_training = conjugates_to_dataframe(df_training)
    df_training = sort_conjugates(df_training)

    return make_pyg_dataset(df_training, node_features, edge_features)


def test_generate_pairwise_data():
    dataset = generate_pairwise_data()
    print(dataset)
