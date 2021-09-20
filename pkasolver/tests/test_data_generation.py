from rdkit import Chem
from pkasolver.data import make_features_dicts, make_nodes
from pkasolver.constants import NODE_FEATURES, EDGE_FEATURES
from pkasolver.data import make_edges_and_attr, make_nodes


def test_features_dicts():
    """Test the generation of the features dict"""
    list_n = ["atomic_number", "formal_charge"]
    list_e = ["bond_type", "is_conjugated"]

    n_feat = make_features_dicts(NODE_FEATURES, list_n)
    e_feat = make_features_dicts(EDGE_FEATURES, list_e)
    assert set(n_feat.keys()) == set(list_n)
    assert len(n_feat.keys()) == len(list_n)
    assert set(e_feat.keys()) == set(list_e)
    assert len(e_feat.keys()) == len(list_e)


def test_dataset():
    """what charges are present in the dataset"""
    from pkasolver.data import preprocess
    import numpy as np

    sdf_filepaths = load_data()
    df = preprocess(sdf_filepaths["Training"])
    charges = []
    for i in range(len(df.index)):
        charge_prot = np.sum(
            [a.GetFormalCharge() for a in df.iloc[i].protonated.GetAtoms()]
        )
        charge_deprot = np.sum(
            [a.GetFormalCharge() for a in df.iloc[i].deprotonated.GetAtoms()]
        )

        # the difference needs to be 1
        assert abs(charge_prot - charge_deprot) == 1
        charges.append(charge_prot)
        charges.append(charge_deprot)

    # NOTE: quite a lot of negative charges
    assert max(charges) == 2
    assert min(charges) == -4


def test_nodes():
    """Test the conversion of mol to nodes with feature subset"""
    list_n = ["atomic_number", "formal_charge"]

    n_feat = make_features_dicts(NODE_FEATURES, list_n)

    mol = Chem.MolFromSmiles("Cc1ccccc1")
    nodes = make_nodes(mol, 1, n_feat)
    for xi in nodes:
        assert tuple(xi.numpy()) == (6.0, 0.0)

    list_n = ["atomic_number", "formal_charge", "marvin_atom"]

    n_feat = make_features_dicts(NODE_FEATURES, list_n)
    nodes = make_nodes(mol, 1, n_feat)
    assert tuple(nodes[1].numpy()) == (6.0, 0.0, 1.0)


def test_edges_generation():
    import torch

    """ Test the conversion of edge indizes and features with feature subset """
    list_e = ["bond_type", "is_conjugated"]

    e_feat = make_features_dicts(EDGE_FEATURES, list_e)

    mol = Chem.MolFromSmiles("Cc1ccccc1")
    edge_index, edge_attr = make_edges_and_attr(mol, e_feat)
    assert torch.equal(
        edge_index,
        torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 1],
                [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 1, 6],
            ]
        ),
    )
    assert torch.equal(
        edge_attr,
        torch.tensor(
            [
                [1.0000, 0.0000],
                [1.0000, 0.0000],
                [1.5000, 1.0000],
                [1.5000, 1.0000],
                [1.5000, 1.0000],
                [1.5000, 1.0000],
                [1.5000, 1.0000],
                [1.5000, 1.0000],
                [1.5000, 1.0000],
                [1.5000, 1.0000],
                [1.5000, 1.0000],
                [1.5000, 1.0000],
                [1.5000, 1.0000],
                [1.5000, 1.0000],
            ]
        ),
    )


def load_data() -> dict:

    """Helper function loading the raw dataset"""

    base = "data/Baltruschat"
    sdf_filepath_training = f"{base}/combined_training_datasets_unique.sdf"
    sdf_filepath_novartis = f"{base}/novartis_cleaned_mono_unique_notraindata.sdf"
    sdf_filepath_AvLiLuMoVe = f"{base}/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf"

    datasets = {
        "Training": sdf_filepath_training,
        "Novartis": sdf_filepath_novartis,
        "AvLiLuMoVe": sdf_filepath_AvLiLuMoVe,
    }
    return datasets


def test_use_dataset_for_node_generation():
    """Test that the training dataset can be generated and that prot/deprot are different molecules"""
    from pkasolver.data import preprocess
    import torch

    sdf_filepaths = load_data()
    df = preprocess(sdf_filepaths["Training"])
    assert df.iloc[0].pKa == 6.21
    assert int(df.iloc[0].marvin_atom) == 10
    assert df.iloc[0].smiles == "Brc1c(NC2CC2)nc(C2CC2)nc1N1CCCCCC1"

    list_n = ["atomic_number", "formal_charge"]
    n_feat = make_features_dicts(NODE_FEATURES, list_n)

    n_prot = make_nodes(df.iloc[0].protonated, df.iloc[0].marvin_atom, n_feat)
    print(n_prot)
    n_dep = make_nodes(df.iloc[0].deprotonated, df.iloc[0].marvin_atom, n_feat)
    print(n_dep)
    assert torch.equal(n_prot, n_dep) == False


def test_generate_data_intances():
    """Test that data classes instances are created correctly"""
    from pkasolver.data import (
        preprocess,
        mol_to_single_mol_data,
        mol_to_paired_mol_data,
    )
    import torch

    ############
    mol_idx = 0
    ############
    sdf_filepaths = load_data()
    df = preprocess(sdf_filepaths["Training"])
    assert df.iloc[mol_idx].pKa == 6.21
    assert int(df.iloc[mol_idx].marvin_atom) == 10
    assert df.iloc[mol_idx].smiles == "Brc1c(NC2CC2)nc(C2CC2)nc1N1CCCCCC1"

    list_n = ["atomic_number", "formal_charge"]
    n_feat = make_features_dicts(NODE_FEATURES, list_n)
    list_e = ["bond_type", "is_conjugated"]
    e_feat = make_features_dicts(EDGE_FEATURES, list_e)

    d1, charge1 = mol_to_single_mol_data(df.iloc[mol_idx], n_feat, e_feat, "protonated")
    d2, charge2 = mol_to_single_mol_data(
        df.iloc[mol_idx], n_feat, e_feat, "deprotonated"
    )
    assert charge1 == 1
    assert charge2 == 0

    d3 = mol_to_paired_mol_data(
        df.iloc[mol_idx],
        n_feat,
        e_feat,
    )
    # all of them have the same number of nodes
    assert d1.num_nodes == d2.num_nodes == len(d3.x_p) == len(d3.x_d)
    # but different node features
    assert torch.equal(d1.x, d2.x) is False
    # they have the same connection table
    assert torch.equal(d1.edge_index, d2.edge_index)
    # but different edge features (NOTE: In the case of this molecule edge attr are the same)
    assert torch.equal(d1.edge_attr, d2.edge_attr) is True

    # Try a new molecule
    ############
    mol_idx = 1429
    ############

    d1, charge1 = mol_to_single_mol_data(df.iloc[mol_idx], n_feat, e_feat, "protonated")
    d2, charge2 = mol_to_single_mol_data(
        df.iloc[mol_idx], n_feat, e_feat, "deprotonated"
    )
    d3 = mol_to_paired_mol_data(
        df.iloc[mol_idx],
        n_feat,
        e_feat,
    )
    print(df.iloc[mol_idx].smiles)
    assert charge1 == 1
    assert charge2 == 0

    # all of them have the same number of nodes
    assert d1.num_nodes == d2.num_nodes == len(d3.x_p) == len(d3.x_d)
    # but different node features
    assert torch.equal(d1.x, d2.x) is False
    # they have the same connection table
    assert torch.equal(d1.edge_index, d2.edge_index)
    # but different edge features (NOTE: In the case of this molecule edge attr are the same)
    assert torch.equal(d1.edge_attr, d2.edge_attr) is True


def test_generate_dataset():
    """Test that data classes instances are created correctly"""
    from pkasolver.data import (
        preprocess,
        mol_to_single_mol_data,
        mol_to_paired_mol_data,
        make_pyg_dataset_based_on_charge,
        make_pyg_dataset_based_on_number_of_hydrogens,
    )

    # setupt dataframe and features
    sdf_filepaths = load_data()
    df = preprocess(sdf_filepaths["Training"])
    list_n = ["atomic_number", "formal_charge"]
    n_feat = make_features_dicts(NODE_FEATURES, list_n)
    list_e = ["bond_type", "is_conjugated"]
    e_feat = make_features_dicts(EDGE_FEATURES, list_e)
    # start with generating datasets based on charge

    # generated PairedData set
    make_pyg_dataset_based_on_charge(df, list_n, list_e, paired=True)
    # generated single Data set
    make_pyg_dataset_based_on_charge(df, list_n, list_e, paired=False)

    # start with generating datasets based on hydrogen count

    # generated PairedData set
    make_pyg_dataset_based_on_number_of_hydrogens(df, list_n, list_e, paired=True)
    # generated single Data set
    make_pyg_dataset_based_on_number_of_hydrogens(df, list_n, list_e, paired=False)
