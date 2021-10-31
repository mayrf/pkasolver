from pkasolver.chem import create_conjugate
from pkasolver.data import (
    _generate_normalized_descriptors,
    load_data,
    make_features_dicts,
    preprocess,
)
from rdkit import Chem
from pkasolver.chem import get_descriptors_from_mol
from pkasolver.data import import_sdf


def test_conjugates():
    from pkasolver.data import import_sdf

    sdf_filepaths = load_data()
    df = import_sdf(sdf_filepaths["Training"])

    for i in range(len(df.index)):
        mol_unchanged = Chem.Mol(df.ROMol[i])
        index = int(df.marvin_atom[i])
        pka = float(df.marvin_pKa[i])
        mol_new = create_conjugate(mol_unchanged, index, pka)
        print(
            mol_unchanged.GetNumHeavyAtoms()
            - mol_unchanged.GetNumAtoms(onlyExplicit=False),
            mol_new.GetNumHeavyAtoms() - mol_new.GetNumAtoms(onlyExplicit=False),
        )
        assert Chem.MolToSmiles(mol_unchanged) != Chem.MolToSmiles(mol_new)


def test_descriptors():
    from pkasolver.chem import get_nr_of_descriptors
    import torch.nn.functional as F
    import torch

    sdf_filepaths = load_data()
    df = import_sdf(sdf_filepaths["Training"])
    smiles = df["smiles"][0]
    print(smiles)
    m = Chem.MolFromSmiles(smiles)
    s = get_descriptors_from_mol(m)
    assert len(s) == 208
    assert len(s) == get_nr_of_descriptors()
    print(torch.tensor(s))
    s_norm = F.normalize(torch.tensor(s), dim=0).tolist()
    print(s_norm)
    torch.tensor([s_norm])


def test_descriptors_fn():
    from pkasolver.constants import NODE_FEATURES, EDGE_FEATURES
    import torch

    ############
    mol_idx = 0
    ############
    sdf_filepaths = load_data()
    df = preprocess(sdf_filepaths["Literature"])

    list_n = ["element", "formal_charge"]
    list_e = ["bond_type", "is_conjugated"]

    selected_node_features = make_features_dicts(NODE_FEATURES, list_n)
    selected_edge_features = make_features_dicts(EDGE_FEATURES, list_e)
    descriptors_p, descriptors_d = _generate_normalized_descriptors(
        df,
        selected_edge_features=selected_edge_features,
        selected_node_features=selected_node_features,
    )
    print(torch.tensor(descriptors_d[0]))
    print(descriptors_d[0])
