import numpy as np
from pkasolver.query import _get_ionization_indices, calculate_microstate_pka_values
from rdkit import Chem
import pytest, os

input = "pkasolver/tests/testdata/00_chembl_subset.sdf"
mollist = []
with open(input, "rb") as fh:
    suppl = Chem.ForwardSDMolSupplier(fh, removeHs=True)
    for i, mol in enumerate(suppl):
        mollist.append(mol)

@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_generate_query():
    from pkasolver.query import QueryModel
    q = QueryModel()


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_predict():
    from pkasolver.constants import EDGE_FEATURES, NODE_FEATURES
    from pkasolver.data import make_features_dicts, mol_to_paired_mol_data
    from pkasolver.ml import dataset_to_dataloader
    from pkasolver.query import QueryModel

    node_feat_list = [
        "element",
        "formal_charge",
        "hybridization",
        "total_num_Hs",
        "aromatic_tag",
        "total_valence",
        "total_degree",
        "is_in_ring",
        "reaction_center",
        "smarts",
    ]

    edge_feat_list = ["bond_type", "is_conjugated", "rotatable"]
    query_model = QueryModel()

    # make dicts from selection list to be used in the processing step
    selected_node_features = make_features_dicts(NODE_FEATURES, node_feat_list)
    selected_edge_features = make_features_dicts(EDGE_FEATURES, edge_feat_list)
    prot = Chem.MolFromSmiles(
        "Nc1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc([NH3+])c5ccccc54)c3)c2)c2ccccc12"
    )
    deprot = Chem.MolFromSmiles(
        "Nc1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc(N)c5ccccc54)c3)c2)c2ccccc12"
    )
    m = mol_to_paired_mol_data(
        prot, deprot, 21, selected_node_features, selected_edge_features,
    )
    loader = dataset_to_dataloader([m], 1, shuffle=False)
    assert np.isclose(query_model.predict_pka_value(loader)[0], 1.7956476402282715)

    deprot = Chem.MolFromSmiles(
        "[NH-]c1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc(N)c5ccccc54)c3)c2)c2ccccc12"
    )
    prot = Chem.MolFromSmiles(
        "Nc1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc(N)c5ccccc54)c3)c2)c2ccccc12"
    )
    m = mol_to_paired_mol_data(
        prot, deprot, 0, selected_node_features, selected_edge_features,
    )
    loader = dataset_to_dataloader([m], 1, shuffle=False)
    assert np.isclose(query_model.predict_pka_value(loader)[0], 11.906086196899414)

    # https://en.wikipedia.org/wiki/Acetic_acid
    deprot = Chem.MolFromSmiles("CC(=O)[O-]")
    prot = Chem.MolFromSmiles("CC(=O)O")

    idx = _get_ionization_indices([deprot, prot], prot)[0]
    m = mol_to_paired_mol_data(
        prot, deprot, idx, selected_node_features, selected_edge_features,
    )
    loader = dataset_to_dataloader([m], 1, shuffle=False)
    assert np.isclose(query_model.predict_pka_value(loader)[0], 4.618539295196533)
    # https://www.masterorganicchemistry.com/2017/04/18/basicity-of-amines-and-pkah/
    deprot = Chem.MolFromSmiles("c1ccncc1")
    prot = Chem.MolFromSmiles("c1cc[nH+]cc1")
    idx = _get_ionization_indices([deprot, prot], prot)[0]
    m = mol_to_paired_mol_data(
        prot, deprot, idx, selected_node_features, selected_edge_features,
    )
    loader = dataset_to_dataloader([m], 1, shuffle=False)
    assert np.isclose(query_model.predict_pka_value(loader)[0], 5.217108573913574)

    # https://www.masterorganicchemistry.com/2017/04/18/basicity-of-amines-and-pkah/
    deprot = Chem.MolFromSmiles("C1CCNCC1")
    prot = Chem.MolFromSmiles("C1CC[NH2+]CC1")
    idx = _get_ionization_indices([deprot, prot], prot)[0]
    m = mol_to_paired_mol_data(
        prot, deprot, idx, selected_node_features, selected_edge_features,
    )
    loader = dataset_to_dataloader([m], 1, shuffle=False)
    assert np.isclose(query_model.predict_pka_value(loader)[0], 11.142233619689941)


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_piperidine():
    # https://www.masterorganicchemistry.com/2017/04/18/basicity-of-amines-and-pkah/
    mol = Chem.MolFromSmiles("C1CCNCC1")
    protonation_states = calculate_microstate_pka_values(mol, only_dimorphite=True)
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert Chem.MolToSmiles(state.protonated_mol) == "C1CC[NH2+]CC1"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "C1CCNCC1"
    print(state.pka)
    assert np.isclose(state.pka, 11.142233428955079)
    assert state.reaction_center_idx == 3

    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    assert len(molpairs) == 1
    protonation_state = 0
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert Chem.MolToSmiles(state.protonated_mol) == "C1CC[NH2+]CC1"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "C1CCNCC1"
    assert np.isclose(state.pka, 11.142233428955079)
    assert state.reaction_center_idx == 3


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_pyridine():
    # https://www.masterorganicchemistry.com/2017/04/18/basicity-of-amines-and-pkah/
    mol = Chem.MolFromSmiles("C1=CC=NC=C1")
    protonation_states = calculate_microstate_pka_values(mol, only_dimorphite=True)
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert Chem.MolToSmiles(state.protonated_mol) == "c1cc[nH+]cc1"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "c1ccncc1"
    assert np.isclose(state.pka, 5.217108669281006)
    assert np.isclose(state.pka_stddev, 0.14740426758793942)
    assert state.reaction_center_idx == 3


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_acetic_acid():
    # https://en.wikipedia.org/wiki/Acetic_acid
    mol = Chem.MolFromSmiles("CC(=O)[O-]")
    protonation_states = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert Chem.MolToSmiles(state.protonated_mol) == "CC(=O)O"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "CC(=O)[O-]"
    assert np.isclose(state.pka, 4.618539333343506)
    assert np.isclose(state.pka_stddev, 0.19455726845025095)
    assert state.reaction_center_idx == 3

    ################################################
    protonation_states = calculate_microstate_pka_values(mol, only_dimorphite=True)
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert Chem.MolToSmiles(state.protonated_mol) == "CC(=O)O"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "CC(=O)[O-]"
    assert np.isclose(state.pka, 4.618539333343506)
    assert np.isclose(state.pka_stddev, 0.19455726845025095)
    assert state.reaction_center_idx == 3


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_fumaric_acid():
    # https://www.waters.com/nextgen/ca/en/library/application-notes/2020/analysis-of-organic-acids-using-a-mixed-mode-lc-column-and-an-acquity-qda-mass-detector.html
    mol = Chem.MolFromSmiles("O=C(O)/C=C/C(=O)O")
    protonation_states = calculate_microstate_pka_values(mol)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    assert len(protonation_states) == 2
    protonation_state = 0
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )

    assert Chem.MolToSmiles(state.protonated_mol) == "O=C(O)/C=C/C(=O)O"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "O=C([O-])/C=C/C(=O)O"
    assert np.isclose(state.pka, 3.4965454959869384)
    assert np.isclose(state.pka_stddev, 0.2759701082882589)
    assert state.reaction_center_idx == 2 or state.reaction_center_idx == 7
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert Chem.MolToSmiles(state.protonated_mol) == "O=C([O-])/C=C/C(=O)O"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "O=C([O-])/C=C/C(=O)[O-]"
    assert np.isclose(state.pka, 4.7557164001464844)
    assert np.isclose(state.pka_stddev, 0.1737669293869924)
    assert state.reaction_center_idx == 2 or state.reaction_center_idx == 7


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_malic_acid():
    # https://www.waters.com/nextgen/ca/en/library/application-notes/2020/analysis-of-organic-acids-using-a-mixed-mode-lc-column-and-an-acquity-qda-mass-detector.html
    mol = Chem.MolFromSmiles("O=C(O)CC(O)C(=O)O")

    ################################################
    ################################################
    protonation_states = calculate_microstate_pka_values(mol)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")

    assert len(protonation_states) == 3
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert Chem.MolToSmiles(state.protonated_mol) == "O=C(O)CC(O)C(=O)O"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "O=C(O)CC(O)C(=O)[O-]"
    assert np.isclose(state.pka, 3.287195949554443)
    assert np.isclose(state.pka_stddev, 0.26364238715918636)
    assert state.reaction_center_idx == 8
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert Chem.MolToSmiles(state.protonated_mol) == "O=C(O)CC(O)C(=O)[O-]"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "O=C([O-])CC(O)C(=O)[O-]"
    assert np.isclose(state.pka, 4.485187845230103)
    assert np.isclose(state.pka_stddev, 0.25245859525128617)
    assert state.reaction_center_idx == 2
    ################################################
    protonation_state = 2
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )

    assert Chem.MolToSmiles(state.protonated_mol) == "O=C([O-])CC(O)C(=O)[O-]"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "O=C([O-])CC([O-])C(=O)[O-]"
    assert np.isclose(state.pka, 11.76492530822754)
    assert np.isclose(state.pka_stddev, 0.8780547512344341)
    assert state.reaction_center_idx == 5


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_citric_acid():
    # https://www.waters.com/nextgen/ca/en/library/application-notes/2020/analysis-of-organic-acids-using-a-mixed-mode-lc-column-and-an-acquity-qda-mass-detector.html
    mol = Chem.MolFromSmiles("O=C(O)CC(O)(CC(=O)O)C(=O)O")

    ################################################
    ################################################
    protonation_states = calculate_microstate_pka_values(mol)
    assert len(protonation_states) == 4
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")

    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert Chem.MolToSmiles(state.protonated_mol) == "O=C(O)CC(O)(CC(=O)O)C(=O)O"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "O=C(O)CC(O)(CC(=O)O)C(=O)[O-]"
    assert np.isclose(state.pka, 2.9028299236297608)
    assert np.isclose(state.pka_stddev, 0.3295379830348129)
    assert state.reaction_center_idx == 12

    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert Chem.MolToSmiles(state.protonated_mol) == "O=C(O)CC(O)(CC(=O)O)C(=O)[O-]"
    assert (
        Chem.MolToSmiles(state.deprotonated_mol) == "O=C([O-])CC(O)(CC(=O)O)C(=O)[O-]"
    )
    assert np.isclose(state.pka, 3.91278226852417)
    assert np.isclose(state.pka_stddev, 0.18345246870885148)
    assert state.reaction_center_idx == 2 or state.reaction_center_idx == 9

    ################################################
    protonation_state = 2
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert Chem.MolToSmiles(state.protonated_mol) == "O=C([O-])CC(O)(CC(=O)O)C(=O)[O-]"
    assert (
        Chem.MolToSmiles(state.deprotonated_mol)
        == "O=C([O-])CC(O)(CC(=O)[O-])C(=O)[O-]"
    )
    assert np.isclose(state.pka, 4.602454824447632)
    assert np.isclose(state.pka_stddev, 0.22445990740843338)

    assert state.reaction_center_idx == 9 or state.reaction_center_idx == 2
    ################################################
    protonation_state = 3
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )

    assert (
        Chem.MolToSmiles(state.protonated_mol) == "O=C([O-])CC(O)(CC(=O)[O-])C(=O)[O-]"
    )
    assert (
        Chem.MolToSmiles(state.deprotonated_mol)
        == "O=C([O-])CC([O-])(CC(=O)[O-])C(=O)[O-]"
    )
    assert np.isclose(state.pka, 12.459272346496583)
    assert np.isclose(state.pka_stddev, 0.870846000823227)
    assert state.reaction_center_idx == 5


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_ascorbic_acid():
    # https://www.waters.com/nextgen/ca/en/library/application-notes/2020/analysis-of-organic-acids-using-a-mixed-mode-lc-column-and-an-acquity-qda-mass-detector.html
    mol = Chem.MolFromSmiles("C(C(C1C(=C(C(=O)O1)O)O)O)O")

    ################################################
    ################################################
    protonation_states = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    ################################################
    assert len(protonation_states) == 3
    ################################################

    protonation_state = 0
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert Chem.MolToSmiles(state.protonated_mol) == "O=C1OC(C(O)CO)C(O)=C1O"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "O=C1OC(C(O)CO)C([O-])=C1O"
    assert np.isclose(state.pka, 3.4077906036376953)
    assert np.isclose(state.pka_stddev, 0.5258236106503124)
    assert state.reaction_center_idx == 9

    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert Chem.MolToSmiles(state.protonated_mol) == "O=C1OC(C(O)CO)C([O-])=C1O"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "O=C1OC(C([O-])CO)C([O-])=C1O"
    assert np.isclose(state.pka, 11.32576545715332)
    assert np.isclose(state.pka_stddev, 0.4914894942226038)
    assert state.reaction_center_idx == 5
    ################################################
    protonation_state = 2
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert Chem.MolToSmiles(state.protonated_mol) == "O=C1OC(C([O-])CO)C([O-])=C1O"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "O=C1OC(C([O-])C[O-])C([O-])=C1O"
    assert np.isclose(state.pka, 12.565634384155274)
    assert np.isclose(state.pka_stddev, 0.7304218950640419)
    assert state.reaction_center_idx == 7


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_quinic_acid():
    # https://www.waters.com/nextgen/ca/en/library/application-notes/2020/analysis-of-organic-acids-using-a-mixed-mode-lc-column-and-an-acquity-qda-mass-detector.html
    mol = Chem.MolFromSmiles("O=C(O)[C@]1(O)C[C@@H](O)[C@@H](O)[C@H](O)C1")

    ################################################
    ################################################
    protonation_states = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")

    ################################################
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert (
        Chem.MolToSmiles(state.protonated_mol)
        == "O=C(O)[C@]1(O)C[C@@H](O)[C@@H](O)[C@H](O)C1"
    )
    assert (
        Chem.MolToSmiles(state.deprotonated_mol)
        == "O=C([O-])[C@]1(O)C[C@@H](O)[C@@H](O)[C@H](O)C1"
    )
    assert np.isclose(state.pka, 3.7475159549713135)
    assert np.isclose(state.pka_stddev, 0.2717212272942886)
    assert state.reaction_center_idx == 2
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert (
        Chem.MolToSmiles(state.protonated_mol)
        == "O=C([O-])[C@]1(O)C[C@@H](O)[C@@H](O)[C@H](O)C1"
    )
    assert (
        Chem.MolToSmiles(state.deprotonated_mol)
        == "O=C([O-])[C@]1([O-])C[C@@H](O)[C@@H](O)[C@H](O)C1"
    )
    assert np.isclose(state.pka, 12.677606239318848)
    assert np.isclose(state.pka_stddev, 0.8569402490624217)
    assert state.reaction_center_idx == 4
    ################################################


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_cocaine():
    # https://www.waters.com/nextgen/ca/en/library/application-notes/2020/analysis-of-organic-acids-using-a-mixed-mode-lc-column-and-an-acquity-qda-mass-detector.html
    mol = Chem.MolFromSmiles("COC(=O)C1C(OC(=O)C2=CC=CC=C2)CC2CCC1N2C")

    ################################################
    ################################################
    protonation_states = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")

    assert len(protonation_states) == 1
    protonation_state = 0
    state = protonation_states[protonation_state]

    assert (
        Chem.MolToSmiles(state.protonated_mol)
        == "COC(=O)C1C(OC(=O)c2ccccc2)CC2CCC1[NH+]2C"
    )
    assert (
        Chem.MolToSmiles(state.deprotonated_mol)
        == "COC(=O)C1C(OC(=O)c2ccccc2)CC2CCC1N2C"
    )
    assert np.isclose(state.pka, 8.35282917022705)
    assert np.isclose(state.pka_stddev, 0.14094952769973257)
    assert state.reaction_center_idx == 20
    ################################################


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_tyrosine():
    # https://www.vanderbilt.edu/AnS/Chemistry/Rizzo/stuff/AA/AminoAcids.html
    mol = Chem.MolFromSmiles("NC(CC1=CC=C(O)C=C1)C(=O)O")
    ################################################
    ################################################
    protonation_states = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    ################################################
    assert len(protonation_states) == 3
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "[NH3+]C(Cc1ccc(O)cc1)C(=O)O"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "[NH3+]C(Cc1ccc(O)cc1)C(=O)[O-]"
    assert np.isclose(state.pka, 2.3857649517059327)
    assert np.isclose(state.pka_stddev, 0.13495164342999938)
    assert state.reaction_center_idx == 12
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "[NH3+]C(Cc1ccc(O)cc1)C(=O)[O-]"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "NC(Cc1ccc(O)cc1)C(=O)[O-]"
    assert np.isclose(state.pka, 9.06846939086914)
    assert np.isclose(state.pka_stddev, 0.26280510436431653)
    assert state.reaction_center_idx == 0
    ################################################
    protonation_state = 2
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "NC(Cc1ccc(O)cc1)C(=O)[O-]"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "NC(Cc1ccc([O-])cc1)C(=O)[O-]"
    assert np.isclose(state.pka, 10.211171607971192)
    assert np.isclose(state.pka_stddev, 0.16842097091292005)
    assert state.reaction_center_idx == 7
    ################################################


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_aspartic_acid():
    # https://chem.libretexts.org/Courses/University_of_Arkansas_Little_Rock/CHEM_4320_5320%3A_Biochemistry_1/01%3A_Amino_Acids/1.4%3A_Reactions_of_Amino_Acids/1.4.1_Acid-base_Chemistry_of_Amino_Acids
    mol = Chem.MolFromSmiles("N[C@@H](CC(=O)O)C(=O)O")
    ################################################
    ################################################
    protonation_states = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    ################################################
    assert len(protonation_states) == 3
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "[NH3+][C@@H](CC(=O)O)C(=O)O"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "[NH3+][C@@H](CC(=O)O)C(=O)[O-]"
    assert np.isclose(state.pka, 2.2450517988204957)
    assert np.isclose(state.pka_stddev, 0.18708663050404836)
    assert state.reaction_center_idx == 8
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "[NH3+][C@@H](CC(=O)O)C(=O)[O-]"
    assert (
        Chem.MolToSmiles(state.deprotonated_mol) == "[NH3+][C@@H](CC(=O)[O-])C(=O)[O-]"
    )
    assert np.isclose(state.pka, 3.37408652305603)
    assert np.isclose(state.pka_stddev, 0.20881348014705198)
    assert state.reaction_center_idx == 5
    ################################################
    protonation_state = 2
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "[NH3+][C@@H](CC(=O)[O-])C(=O)[O-]"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "N[C@@H](CC(=O)[O-])C(=O)[O-]"
    assert np.isclose(state.pka, 9.942882041931153)
    assert np.isclose(state.pka_stddev, 0.33096213062277385)
    assert state.reaction_center_idx == 0
    ################################################


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_taurin():
    # http://www2.chm.ulaval.ca/gecha/chm1903/4_acide-base/organic_acids.pdf
    mol = Chem.MolFromSmiles("NCCS(=O)(=O)O")
    ################################################
    ################################################
    protonation_states = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    ################################################
    assert len(protonation_states) == 1
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "[NH3+]CCS(=O)(=O)[O-]"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "NCCS(=O)(=O)[O-]"
    assert np.isclose(state.pka, 9.273014373779297)
    assert np.isclose(state.pka_stddev, 0.3919081083766658)
    assert state.reaction_center_idx == 0
    ################################################


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_cysteamine():
    # http://www2.chm.ulaval.ca/gecha/chm1903/4_acide-base/organic_acids.pdf
    mol = Chem.MolFromSmiles("NCCS")
    ################################################
    ################################################
    protonation_states = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    ################################################
    assert len(protonation_states) == 3
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "[NH3+]CC[SH2+]"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "[NH3+]CCS"
    assert np.isclose(state.pka, 2.8358943247795105)
    assert np.isclose(state.pka_stddev, 2.0573288241571532)
    assert state.reaction_center_idx == 3
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "[NH3+]CCS"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "[NH3+]CC[S-]"
    assert np.isclose(state.pka, 8.246431522369384)
    assert np.isclose(state.pka_stddev, 0.4282842336283469)
    assert state.reaction_center_idx == 3
    ################################################
    protonation_state = 2
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "[NH3+]CC[S-]"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "NCC[S-]"
    assert np.isclose(state.pka, 10.842092018127442)
    assert np.isclose(state.pka_stddev, 0.30701055827620577)
    assert state.reaction_center_idx == 0
    ################################################


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_diphonoethane():
    # http://www2.chm.ulaval.ca/gecha/chm1903/4_acide-base/organic_acids.pdf
    mol = Chem.MolFromSmiles("CC(O)(P(=O)(O)O)P(=O)(O)O")
    ################################################
    ################################################
    protonation_states = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    ################################################
    assert len(protonation_states) == 4
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "CC(O)(P(=O)(O)O)P(=O)(O)O"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "CC(O)(P(=O)([O-])O)P(=O)(O)O"
    assert np.isclose(state.pka, 1.6972961282730104)
    assert np.isclose(state.pka_stddev, 0.39150291555536587)
    assert (
        state.reaction_center_idx == 9 or state.reaction_center_idx == 5
    )  # symmetric mol
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "CC(O)(P(=O)([O-])O)P(=O)(O)O"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "CC(O)(P(=O)([O-])O)P(=O)([O-])O"
    assert np.isclose(state.pka, 2.4382746505737303)
    assert np.isclose(state.pka_stddev, 0.2560998882164273)
    assert (
        state.reaction_center_idx == 9 or state.reaction_center_idx == 5
    )  # symmetric mol
    ################################################
    protonation_state = 2
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "CC(O)(P(=O)([O-])O)P(=O)([O-])O"
    assert (
        Chem.MolToSmiles(state.deprotonated_mol) == "CC(O)(P(=O)([O-])[O-])P(=O)([O-])O"
    )
    assert np.isclose(state.pka, 5.742288055419922)
    assert np.isclose(state.pka_stddev, 0.8135513898177413)
    assert state.reaction_center_idx == 6 or state.reaction_center_idx == 10
    ################################################
    protonation_state = 3
    state = protonation_states[protonation_state]

    assert (
        Chem.MolToSmiles(state.protonated_mol) == "CC(O)(P(=O)([O-])[O-])P(=O)([O-])O"
    )
    assert (
        Chem.MolToSmiles(state.deprotonated_mol)
        == "CC(O)(P(=O)([O-])[O-])P(=O)([O-])[O-]"
    )
    assert np.isclose(state.pka, 10.390039710998535)
    assert np.isclose(state.pka_stddev, 0.6905222291846247)
    assert state.reaction_center_idx == 10 or state.reaction_center_idx == 6
    ################################################


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_arginin():
    # https://chem.libretexts.org/Courses/University_of_Arkansas_Little_Rock/CHEM_4320_5320%3A_Biochemistry_1/01%3A_Amino_Acids/1.4%3A_Reactions_of_Amino_Acids/1.4.1_Acid-base_Chemistry_of_Amino_Acids
    mol = Chem.MolFromSmiles("NC(N)=NCCCC(N)C(=O)O")
    ################################################
    ################################################
    protonation_states = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    ################################################
    assert len(protonation_states) == 3
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "NC(N)=[NH+]CCCC([NH3+])C(=O)O"
    assert (
        Chem.MolToSmiles(state.deprotonated_mol) == "NC(N)=[NH+]CCCC([NH3+])C(=O)[O-]"
    )
    assert np.isclose(state.pka, 2.445406141281128)
    assert np.isclose(state.pka_stddev, 0.12240883966818901)
    assert state.reaction_center_idx == 11
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "NC(N)=[NH+]CCCC([NH3+])C(=O)[O-]"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "NC(N)=[NH+]CCCC(N)C(=O)[O-]"
    assert np.isclose(state.pka, 9.563768844604493)
    assert np.isclose(state.pka_stddev, 0.42055264905461387)
    assert state.reaction_center_idx == 8
    ################################################
    protonation_state = 2
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "NC(N)=[NH+]CCCC(N)C(=O)[O-]"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "NC(N)=NCCCC(N)C(=O)[O-]"
    assert np.isclose(state.pka, 11.442546997070313)
    assert np.isclose(state.pka_stddev, 0.31261737113632293)
    assert state.reaction_center_idx == 3
    ################################################


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_thiophenecarboxylicacid():
    # http://www2.chm.ulaval.ca/gecha/chm1903/4_acide-base/organic_acids.pdf
    mol = Chem.MolFromSmiles("O=C(O)C1=CSC=C1")
    ################################################
    ################################################
    protonation_states = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    ################################################
    assert len(protonation_states) == 1
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "O=C(O)c1ccsc1"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "O=C([O-])c1ccsc1"
    assert np.isclose(state.pka, 4.004269208908081)
    assert np.isclose(state.pka_stddev, 0.14052944346087354)
    assert state.reaction_center_idx == 2


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_nitroaniline():
    # http://www2.chm.ulaval.ca/gecha/chm1903/4_acide-base/organic_acids.pdf
    mol = Chem.MolFromSmiles("NC1=CC([N+](=O)[O-])=CC=C1")
    ################################################
    ################################################
    protonation_states = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    ################################################
    assert len(protonation_states) == 2
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "[NH3+]c1cccc([N+](=O)[O-])c1"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "Nc1cccc([N+](=O)[O-])c1"
    assert np.isclose(state.pka, 2.0625481700897215)
    assert np.isclose(state.pka_stddev, 0.40919874942868256)
    assert state.reaction_center_idx == 0
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "Nc1cccc([N+](=O)[O-])c1"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "[NH-]c1cccc([N+](=O)[O-])c1"
    assert np.isclose(state.pka, 12.545295104980468)
    assert np.isclose(state.pka_stddev, 0.47944702638217124)
    assert state.reaction_center_idx == 0


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_benzenesulfinicacid():
    # http://www2.chm.ulaval.ca/gecha/chm1903/4_acide-base/organic_acids.pdf
    mol = Chem.MolFromSmiles("O=S(O)C1=CC=CC=C1")
    ################################################
    ################################################
    protonation_states = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    ################################################
    assert len(protonation_states) == 1
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "O=S(O)c1ccccc1"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "O=S([O-])c1ccccc1"
    assert np.isclose(state.pka, 2.213827681541443)
    assert np.isclose(state.pka_stddev, 0.35222117167807243)
    assert state.reaction_center_idx == 2


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_bromobenzoicacid():
    # http://www2.chm.ulaval.ca/gecha/chm1903/4_acide-base/organic_acids.pdf
    mol = Chem.MolFromSmiles("O=C(O)C1=CC(Br)=CC=C1")
    ################################################
    ################################################
    protonation_states = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    ################################################
    assert len(protonation_states) == 1
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "O=C(O)c1cccc(Br)c1"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "O=C([O-])c1cccc(Br)c1"
    assert np.isclose(state.pka, 3.688530502319336)
    assert np.isclose(state.pka_stddev, 0.11760715932181545)
    assert state.reaction_center_idx == 2


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_benzaldehyde():
    # http://www2.chm.ulaval.ca/gecha/chm1903/4_acide-base/organic_acids.pdf
    mol = Chem.MolFromSmiles("O=CC1=CC=CC=C1")
    ################################################
    ################################################
    protonation_states = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    ################################################
    assert len(protonation_states) == 0
    ################################################


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_mol_00():
    # 00 Chembl molecule
    mol = mollist[0]

    protonation_states = calculate_microstate_pka_values(mol)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert (
        Chem.MolToSmiles(state.protonated_mol)
        == "[NH3+]c1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc([NH3+])c5ccccc54)c3)c2)c2ccccc12"
    )
    assert (
        Chem.MolToSmiles(state.deprotonated_mol)
        == "Nc1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc([NH3+])c5ccccc54)c3)c2)c2ccccc12"
    )
    assert np.isclose(state.pka, 1.622351369857788)
    assert np.isclose(state.pka_stddev, 0.38105998542367125)
    assert state.reaction_center_idx == 21 or state.reaction_center_idx == 0
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]

    assert (
        Chem.MolToSmiles(state.protonated_mol)
        == "Nc1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc([NH3+])c5ccccc54)c3)c2)c2ccccc12"
    )
    assert (
        Chem.MolToSmiles(state.deprotonated_mol)
        == "Nc1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc(N)c5ccccc54)c3)c2)c2ccccc12"
    )
    assert np.isclose(state.pka, 2.079871563911438)
    assert np.isclose(state.pka_stddev, 0.48864702887959693)
    assert state.reaction_center_idx == 21 or state.reaction_center_idx == 0
    ################################################
    protonation_state = 2
    state = protonation_states[protonation_state]
    assert (
        Chem.MolToSmiles(state.protonated_mol)
        == "Nc1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc(N)c5ccccc54)c3)c2)c2ccccc12"
    )
    assert (
        Chem.MolToSmiles(state.deprotonated_mol)
        == "[NH-]c1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc(N)c5ccccc54)c3)c2)c2ccccc12"
    )

    assert np.isclose(state.pka, 11.906086082458495)
    assert np.isclose(state.pka_stddev, 0.5356310747117345)
    assert state.reaction_center_idx == 21 or state.reaction_center_idx == 0
    ################################################
    protonation_state = 3
    state = protonation_states[protonation_state]

    assert (
        Chem.MolToSmiles(state.protonated_mol)
        == "[NH-]c1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc(N)c5ccccc54)c3)c2)c2ccccc12"
    )
    assert (
        Chem.MolToSmiles(state.deprotonated_mol)
        == "[NH-]c1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc([NH-])c5ccccc54)c3)c2)c2ccccc12"
    )

    assert np.isclose(state.pka, 12.231420288085937)
    assert np.isclose(state.pka_stddev, 0.5061111726042535)
    assert state.reaction_center_idx == 21 or state.reaction_center_idx == 0

    print("#####################################################")
    print("#####################################################")


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_mol_14():
    # 14th Chembl molecule
    mol = mollist[14]
    protonation_states = calculate_microstate_pka_values(mol, only_dimorphite=False)
    assert (len(protonation_states)) == 1
    # assert len(molpairs) ==
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]

    assert (
        Chem.MolToSmiles(state.protonated_mol)
        == "O[NH2+]C1c2cc(O)c(O)cc2-c2cc(O)c(O)cc21"
    )
    assert (
        Chem.MolToSmiles(state.deprotonated_mol) == "ONC1c2cc(O)c(O)cc2-c2cc(O)c(O)cc21"
    )
    assert np.isclose(state.pka, 3.794149408340454)
    assert np.isclose(state.pka_stddev, 0.4318209957017381)
    assert state.reaction_center_idx == 1


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_mol_20():
    # 20th Chembl molecule
    mol = mollist[20]
    protonation_states = calculate_microstate_pka_values(mol)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]
    assert (
        Chem.MolToSmiles(state.protonated_mol)
        == "CC1(C)C2=CCC3C4CC[C@H](OC5CC5)[C@@]4(C)CCC3[C@@]2(C)CC[C@@H]1O"
    )
    assert (
        Chem.MolToSmiles(state.deprotonated_mol)
        == "CC1(C)C2=CCC3C4CC[C@H](OC5CC5)[C@@]4(C)CCC3[C@@]2(C)CC[C@@H]1[O-]"
    )

    assert np.isclose(state.pka, 12.073209114074707)
    assert np.isclose(state.pka_stddev, 0.8541469800719144)
    assert state.reaction_center_idx == 25


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_mol_58():
    # 58th Chembl molecule
    mol = mollist[58]
    protonation_states = calculate_microstate_pka_values(mol)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]
    assert (
        Chem.MolToSmiles(state.protonated_mol)
        == "CCCCCC[NH+]1CC[NH+]2CC(c3ccccc3)c3ccccc3C2C1"
    )
    assert (
        Chem.MolToSmiles(state.deprotonated_mol)
        == "CCCCCCN1CC[NH+]2CC(c3ccccc3)c3ccccc3C2C1"
    )

    assert np.isclose(state.pka, 5.265505733489991)
    assert np.isclose(state.pka_stddev, 0.6582911877362471)
    assert state.reaction_center_idx == 6
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]
    assert (
        Chem.MolToSmiles(state.protonated_mol)
        == "CCCCCCN1CC[NH+]2CC(c3ccccc3)c3ccccc3C2C1"
    )
    assert (
        Chem.MolToSmiles(state.deprotonated_mol)
        == "CCCCCCN1CCN2CC(c3ccccc3)c3ccccc3C2C1"
    )

    assert np.isclose(state.pka, 7.988929748535156)
    assert np.isclose(state.pka_stddev, 0.32012542241143005)
    assert state.reaction_center_idx == 9


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_mol_59():
    # 59th Chembl molecule
    mol = mollist[59]
    protonation_states = calculate_microstate_pka_values(mol)
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    ################################################
    assert len(protonation_states) == 1
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]
    assert (
        Chem.MolToSmiles(state.protonated_mol)
        == "CC/C(=C(\c1ccc(I)cc1)c1ccc(OCC[NH+](C)C)cc1)c1ccccc1"
    )
    assert (
        Chem.MolToSmiles(state.deprotonated_mol)
        == "CC/C(=C(\c1ccc(I)cc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"
    )
    assert np.isclose(state.pka, 8.490106735229492)
    assert np.isclose(state.pka_stddev, 0.13807002920880312)
    assert state.reaction_center_idx == 18


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_mol_62():
    # 62th Chembl molecule
    mol = mollist[62]
    protonation_states = calculate_microstate_pka_values(mol)
    ################################################
    assert len(protonation_states) == 2
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]
    assert (
        Chem.MolToSmiles(state.protonated_mol)
        == "Cc1cc(CCCCCOc2ccc(-c3[nH+]c(C)c(C)o3)cc2)o[nH+]1"
    )
    assert (
        Chem.MolToSmiles(state.deprotonated_mol)
        == "Cc1cc(CCCCCOc2ccc(-c3nc(C)c(C)o3)cc2)o[nH+]1"
    )
    assert np.isclose(state.pka, 1.0180853271484376)
    assert np.isclose(state.pka_stddev, 0.2517946559772521)
    assert state.reaction_center_idx == 15
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]
    assert (
        Chem.MolToSmiles(state.protonated_mol)
        == "Cc1cc(CCCCCOc2ccc(-c3nc(C)c(C)o3)cc2)o[nH+]1"
    )
    assert (
        Chem.MolToSmiles(state.deprotonated_mol)
        == "Cc1cc(CCCCCOc2ccc(-c3nc(C)c(C)o3)cc2)on1"
    )
    assert np.isclose(state.pka, 2.523374557495117)
    assert np.isclose(state.pka_stddev, 0.7158138563210654)
    assert state.reaction_center_idx == 24


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Needs pretrained GNN models",
)
def test_mol_70():
    # 70th Chembl molecule
    mol = mollist[70]
    protonation_states = calculate_microstate_pka_values(mol)
    ################################################
    assert len(protonation_states) == 2
    ################################################
    print("################################")
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        print(
            Chem.MolToSmiles(state.protonated_mol),
            Chem.MolToSmiles(state.deprotonated_mol),
        )
        print(state.pka)
    print("################################")
    ################################################
    ################################################
    protonation_state = 0
    state = protonation_states[protonation_state]
    assert (
        Chem.MolToSmiles(state.protonated_mol)
        == "Oc1ccc(/C(=C(/c2ccc(O)cc2)C(F)(F)F)C(F)(F)F)cc1"
    )
    assert (
        Chem.MolToSmiles(state.deprotonated_mol)
        == "[O-]c1ccc(/C(=C(/c2ccc(O)cc2)C(F)(F)F)C(F)(F)F)cc1"
    )
    assert np.isclose(state.pka, 8.504621658325195)
    assert np.isclose(state.pka_stddev, 0.21493671292423686)
    assert state.reaction_center_idx == 0 or state.reaction_center_idx == 11

    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]
    assert (
        Chem.MolToSmiles(state.protonated_mol)
        == "[O-]c1ccc(/C(=C(/c2ccc(O)cc2)C(F)(F)F)C(F)(F)F)cc1"
    )
    assert (
        Chem.MolToSmiles(state.deprotonated_mol)
        == "[O-]c1ccc(/C(=C(/c2ccc([O-])cc2)C(F)(F)F)C(F)(F)F)cc1"
    )
    assert np.isclose(state.pka, 9.089583587646484)
    assert np.isclose(state.pka_stddev, 0.17076031530766353)
    assert state.reaction_center_idx == 11 or state.reaction_center_idx == 0


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Needs pretrained GNN models",
)
def test_visualizing():
    from pkasolver.query import draw_pka_reactions, draw_pka_map

    mol = mollist[70]
    protonation_states = calculate_microstate_pka_values(mol)
    draw_pka_reactions(protonation_states)
    draw_pka_map(protonation_states)
