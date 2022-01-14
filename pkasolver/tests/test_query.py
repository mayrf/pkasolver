import numpy as np
from pkasolver.query import _get_ionization_indices, calculate_microstate_pka_values
from rdkit import Chem

input = "pkasolver/tests/testdata/00_chembl_subset.sdf"
mollist = []
with open(input, "rb") as fh:
    suppl = Chem.ForwardSDMolSupplier(fh, removeHs=True)
    for i, mol in enumerate(suppl):
        mollist.append(mol)


def test_generate_query():
    from pkasolver.query import QueryModel

    q = QueryModel()


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
    assert np.isclose(query_model.predict_pka_value(loader)[0], 1.9562606943978205)

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
    assert np.isclose(query_model.predict_pka_value(loader)[0], 11.866702397664389)

    # https://en.wikipedia.org/wiki/Acetic_acid
    deprot = Chem.MolFromSmiles("CC(=O)[O-]")
    prot = Chem.MolFromSmiles("CC(=O)O")

    idx = _get_ionization_indices([deprot, prot], prot)[0]
    m = mol_to_paired_mol_data(
        prot, deprot, idx, selected_node_features, selected_edge_features,
    )
    loader = dataset_to_dataloader([m], 1, shuffle=False)
    assert np.isclose(query_model.predict_pka_value(loader)[0], 4.71059677335951)
    # https://www.masterorganicchemistry.com/2017/04/18/basicity-of-amines-and-pkah/
    deprot = Chem.MolFromSmiles("c1ccncc1")
    prot = Chem.MolFromSmiles("c1cc[nH+]cc1")
    idx = _get_ionization_indices([deprot, prot], prot)[0]
    m = mol_to_paired_mol_data(
        prot, deprot, idx, selected_node_features, selected_edge_features,
    )
    loader = dataset_to_dataloader([m], 1, shuffle=False)
    assert np.isclose(query_model.predict_pka_value(loader)[0], 5.307672288682726)

    # https://www.masterorganicchemistry.com/2017/04/18/basicity-of-amines-and-pkah/
    deprot = Chem.MolFromSmiles("C1CCNCC1")
    prot = Chem.MolFromSmiles("C1CC[NH2+]CC1")
    idx = _get_ionization_indices([deprot, prot], prot)[0]
    m = mol_to_paired_mol_data(
        prot, deprot, idx, selected_node_features, selected_edge_features,
    )
    loader = dataset_to_dataloader([m], 1, shuffle=False)
    assert np.isclose(query_model.predict_pka_value(loader)[0], 11.02138582865397)


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
    assert np.isclose(state.pka, 11.02138582865397)
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
    assert np.isclose(state.pka, 11.02138582865397)
    assert state.reaction_center_idx == 3


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
    assert np.isclose(state.pka, 5.307672023773193)
    assert np.isclose(state.pka_stddev, 0.2175417316093904)
    assert state.reaction_center_idx == 3


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
    assert np.isclose(state.pka, 4.710596614413792)
    assert np.isclose(state.pka_stddev, 0.1525030106240527)
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
    assert np.isclose(state.pka, 4.710596614413792)
    assert np.isclose(state.pka_stddev, 0.1525030106240527)
    assert state.reaction_center_idx == 3


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
    assert np.isclose(state.pka, 3.632740444607205)
    assert np.isclose(state.pka_stddev, 0.17687526045421995)
    assert state.reaction_center_idx == 2 or state.reaction_center_idx == 7
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert Chem.MolToSmiles(state.protonated_mol) == "O=C([O-])/C=C/C(=O)O"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "O=C([O-])/C=C/C(=O)[O-]"
    assert np.isclose(state.pka, 4.8247707155015735)
    assert np.isclose(state.pka_stddev, 0.12407228765387628)
    assert state.reaction_center_idx == 2 or state.reaction_center_idx == 7


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
    assert np.isclose(state.pka, 3.220769908693102)
    assert np.isclose(state.pka_stddev, 0.2966082873268137)
    assert state.reaction_center_idx == 8
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert Chem.MolToSmiles(state.protonated_mol) == "O=C(O)CC(O)C(=O)[O-]"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "O=C([O-])CC(O)C(=O)[O-]"
    assert np.isclose(state.pka, 4.408264848921034)
    assert np.isclose(state.pka_stddev, 0.26891448572304694)
    assert state.reaction_center_idx == 2
    ################################################
    protonation_state = 2
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )

    assert Chem.MolToSmiles(state.protonated_mol) == "O=C([O-])CC(O)C(=O)[O-]"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "O=C([O-])CC([O-])C(=O)[O-]"
    assert np.isclose(state.pka, 12.214655452304417)
    assert np.isclose(state.pka_stddev, 0.9703170605601181)
    assert state.reaction_center_idx == 5


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
    assert np.isclose(state.pka, 2.933947377734714)
    assert np.isclose(state.pka_stddev, 0.2241786070538232)
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
    assert np.isclose(state.pka, 3.884802050060696)
    assert np.isclose(state.pka_stddev, 0.22380544232649632)
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
    assert np.isclose(state.pka, 4.3484864764743385)
    assert np.isclose(state.pka_stddev, 0.17843601915633509)

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
    assert np.isclose(state.pka, 12.627780702379015)
    assert np.isclose(state.pka_stddev, 0.8613784936107726)
    assert state.reaction_center_idx == 5


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
    assert np.isclose(state.pka, 3.392546203401354)
    assert np.isclose(state.pka_stddev, 0.4879326504218696)
    assert state.reaction_center_idx == 9

    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert Chem.MolToSmiles(state.protonated_mol) == "O=C1OC(C(O)CO)C([O-])=C1O"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "O=C1OC(C([O-])CO)C([O-])=C1O"
    assert np.isclose(state.pka, 11.445924123128256)
    assert np.isclose(state.pka_stddev, 0.7339998881296113)
    assert state.reaction_center_idx == 5
    ################################################
    protonation_state = 2
    state = protonation_states[protonation_state]
    print(
        Chem.MolToSmiles(state.protonated_mol), Chem.MolToSmiles(state.deprotonated_mol)
    )
    assert Chem.MolToSmiles(state.protonated_mol) == "O=C1OC(C([O-])CO)C([O-])=C1O"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "O=C1OC(C([O-])C[O-])C([O-])=C1O"
    assert np.isclose(state.pka, 12.827690018547905)
    assert np.isclose(state.pka_stddev, 0.7134137863328631)
    assert state.reaction_center_idx == 7


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
    assert np.isclose(state.pka, 3.8251733779907227)
    assert np.isclose(state.pka_stddev, 0.2510346828149071)
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
    assert np.isclose(state.pka, 12.723742061191135)
    assert np.isclose(state.pka_stddev, 0.6929774350816233)
    assert state.reaction_center_idx == 4
    ################################################


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
    assert np.isclose(state.pka, 8.15889220767551)
    assert np.isclose(state.pka_stddev, 0.1595246303628425)
    assert state.reaction_center_idx == 20
    ################################################


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
    assert np.isclose(state.pka, 2.3078861236572266)
    assert np.isclose(state.pka_stddev, 0.1224474735260068)
    assert state.reaction_center_idx == 12
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "[NH3+]C(Cc1ccc(O)cc1)C(=O)[O-]"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "NC(Cc1ccc(O)cc1)C(=O)[O-]"
    assert np.isclose(state.pka, 9.048761791653103)
    assert np.isclose(state.pka_stddev, 0.21774343020349576)
    assert state.reaction_center_idx == 0
    ################################################
    protonation_state = 2
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "NC(Cc1ccc(O)cc1)C(=O)[O-]"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "NC(Cc1ccc([O-])cc1)C(=O)[O-]"
    assert np.isclose(state.pka, 10.180555873446995)
    assert np.isclose(state.pka_stddev, 0.17844083242750147)
    assert state.reaction_center_idx == 7
    ################################################


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
    assert np.isclose(state.pka, 2.3238737848069935)
    assert np.isclose(state.pka_stddev, 0.13488637306950063)
    assert state.reaction_center_idx == 8
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "[NH3+][C@@H](CC(=O)O)C(=O)[O-]"
    assert (
        Chem.MolToSmiles(state.deprotonated_mol) == "[NH3+][C@@H](CC(=O)[O-])C(=O)[O-]"
    )
    assert np.isclose(state.pka, 3.436119318008423)
    assert np.isclose(state.pka_stddev, 0.18921565287985567)
    assert state.reaction_center_idx == 5
    ################################################
    protonation_state = 2
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "[NH3+][C@@H](CC(=O)[O-])C(=O)[O-]"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "N[C@@H](CC(=O)[O-])C(=O)[O-]"
    assert np.isclose(state.pka, 9.819878578186035)
    assert np.isclose(state.pka_stddev, 0.5142168741173878)
    assert state.reaction_center_idx == 0
    ################################################


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
    assert np.isclose(state.pka, 9.401536623636881)
    assert np.isclose(state.pka_stddev, 0.3991919039325937)
    assert state.reaction_center_idx == 0
    ################################################


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
    assert np.isclose(state.pka, 1.6365958452224731)
    assert np.isclose(state.pka_stddev, 1.282233764720526)
    assert state.reaction_center_idx == 3
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "[NH3+]CCS"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "[NH3+]CC[S-]"
    assert np.isclose(state.pka, 8.010329405466715)
    assert np.isclose(state.pka_stddev, 0.5729369141366224)
    assert state.reaction_center_idx == 3
    ################################################
    protonation_state = 2
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "[NH3+]CC[S-]"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "NCC[S-]"
    assert np.isclose(state.pka, 10.831065601772732)
    assert np.isclose(state.pka_stddev, 0.15334697970028535)
    assert state.reaction_center_idx == 0
    ################################################


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
    assert np.isclose(state.pka, 1.6844406922658284)
    assert np.isclose(state.pka_stddev, 0.3869652303746851)
    assert (
        state.reaction_center_idx == 9 or state.reaction_center_idx == 5
    )  # symmetric mol
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "CC(O)(P(=O)([O-])O)P(=O)(O)O"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "CC(O)(P(=O)([O-])O)P(=O)([O-])O"
    assert np.isclose(state.pka, 2.498790899912516)
    assert np.isclose(state.pka_stddev, 0.2309926882414036)
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
    assert np.isclose(state.pka, 6.107084963056776)
    assert np.isclose(state.pka_stddev, 0.4566133944468266)
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
    assert np.isclose(state.pka, 10.910407384236654)
    assert np.isclose(state.pka_stddev, 0.5324594385636086)
    assert state.reaction_center_idx == 10 or state.reaction_center_idx == 6
    ################################################


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
    assert np.isclose(state.pka, 2.3508769141303167)
    assert np.isclose(state.pka_stddev, 0.14892467791735128)
    assert state.reaction_center_idx == 11
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "NC(N)=[NH+]CCCC([NH3+])C(=O)[O-]"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "NC(N)=[NH+]CCCC(N)C(=O)[O-]"
    assert np.isclose(state.pka, 9.587184376186794)
    assert np.isclose(state.pka_stddev, 0.3218740254203085)
    assert state.reaction_center_idx == 8
    ################################################
    protonation_state = 2
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "NC(N)=[NH+]CCCC(N)C(=O)[O-]"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "NC(N)=NCCCC(N)C(=O)[O-]"
    assert np.isclose(state.pka, 11.523349232143826)
    assert np.isclose(state.pka_stddev, 0.24151349002249864)
    assert state.reaction_center_idx == 3
    ################################################


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
    assert np.isclose(state.pka, 3.991380082236396)
    assert np.isclose(state.pka_stddev, 0.12863774219236004)
    assert state.reaction_center_idx == 2


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
    assert np.isclose(state.pka, 2.0475387705696955)
    assert np.isclose(state.pka_stddev, 0.5670104378808754)
    assert state.reaction_center_idx == 0
    ################################################
    protonation_state = 1
    state = protonation_states[protonation_state]

    assert Chem.MolToSmiles(state.protonated_mol) == "Nc1cccc([N+](=O)[O-])c1"
    assert Chem.MolToSmiles(state.deprotonated_mol) == "[NH-]c1cccc([N+](=O)[O-])c1"
    assert np.isclose(state.pka, 12.237251387702095)
    assert np.isclose(state.pka_stddev, 0.5571708488875928)
    assert state.reaction_center_idx == 0


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
    assert np.isclose(state.pka, 1.939287781715393)
    assert np.isclose(state.pka_stddev, 0.31625038924956134)
    assert state.reaction_center_idx == 2


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
    assert np.isclose(state.pka, 3.667213201522827)
    assert np.isclose(state.pka_stddev, 0.10579638958142196)
    assert state.reaction_center_idx == 2


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
    assert np.isclose(state.pka, 1.6297088596555922)
    assert np.isclose(state.pka_stddev, 0.4077335606093385)
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
    assert np.isclose(state.pka, 2.131844719250997)
    assert np.isclose(state.pka_stddev, 0.5786441572654777)
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

    assert np.isclose(state.pka, 11.8667025036282)
    assert np.isclose(state.pka_stddev, 0.3902020704139609)
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

    assert np.isclose(state.pka, 12.217913839552137)
    assert np.isclose(state.pka_stddev, 0.30860616351880715)
    assert state.reaction_center_idx == 21 or state.reaction_center_idx == 0

    print("#####################################################")
    print("#####################################################")


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
    assert np.isclose(state.pka, 3.922899219724867)
    assert np.isclose(state.pka_stddev, 0.6474637184890659)
    assert state.reaction_center_idx == 1


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

    assert np.isclose(state.pka, 12.213901625739204)
    assert np.isclose(state.pka_stddev, 0.7435417816283145)
    assert state.reaction_center_idx == 25


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

    assert np.isclose(state.pka, 5.111536555820042)
    assert np.isclose(state.pka_stddev, 0.4876307405371854)
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

    assert np.isclose(state.pka, 8.384064144558376)
    assert np.isclose(state.pka_stddev, 0.17997652727358723)
    assert state.reaction_center_idx == 9


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
    assert np.isclose(state.pka, 8.446443027920193)
    assert np.isclose(state.pka_stddev, 0.10109291819060486)
    assert state.reaction_center_idx == 18


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
    assert np.isclose(state.pka, 0.9878645141919454)
    assert np.isclose(state.pka_stddev, 0.24017836755055452)
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
    assert np.isclose(state.pka, 2.2003875176111856)
    assert np.isclose(state.pka_stddev, 0.7891234987927331)
    assert state.reaction_center_idx == 24


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
    assert np.isclose(state.pka, 8.421117199791802)
    assert np.isclose(state.pka_stddev, 0.2949680779937196)
    assert state.reaction_center_idx == 0

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
    assert np.isclose(state.pka, 9.098628362019857)
    assert np.isclose(state.pka_stddev, 0.19024451371832968)
    assert state.reaction_center_idx == 11


def test_visualizing():
    from pkasolver.query import draw_pka_reactions, draw_pka_map

    mol = mollist[70]
    protonation_states = calculate_microstate_pka_values(mol)
    draw_pka_reactions(protonation_states)
    draw_pka_map(protonation_states)
