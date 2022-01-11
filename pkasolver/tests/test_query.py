from rdkit import Chem
from pkasolver.query import calculate_microstate_pka_values, get_ionization_indices
import numpy as np

input = "pkasolver/tests/testdata/00_chembl_subset.sdf"
mollist = []
with open(input, "rb") as fh:
    suppl = Chem.ForwardSDMolSupplier(fh, removeHs=True)
    for i, mol in enumerate(suppl):
        mollist.append(mol)


def test_predict():
    from pkasolver.data import (
        make_features_dicts,
        mol_to_paired_mol_data,
    )
    from pkasolver.ml import dataset_to_dataloader, predict_pka_value
    from pkasolver.constants import EDGE_FEATURES, NODE_FEATURES
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
    assert np.isclose(predict_pka_value(query_model.model, loader), 2.39621401)

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
    assert np.isclose(predict_pka_value(query_model.model, loader), 11.28736687)

    # https://en.wikipedia.org/wiki/Acetic_acid
    deprot = Chem.MolFromSmiles("CC(=O)[O-]")
    prot = Chem.MolFromSmiles("CC(=O)O")

    idx = get_ionization_indices([deprot, prot])[0]
    m = mol_to_paired_mol_data(
        prot, deprot, idx, selected_node_features, selected_edge_features,
    )
    loader = dataset_to_dataloader([m], 1, shuffle=False)
    assert np.isclose(predict_pka_value(query_model.model, loader), 4.81062174)
    # https://www.masterorganicchemistry.com/2017/04/18/basicity-of-amines-and-pkah/
    deprot = Chem.MolFromSmiles("c1ccncc1")
    prot = Chem.MolFromSmiles("c1cc[nH+]cc1")
    idx = get_ionization_indices([deprot, prot])[0]
    m = mol_to_paired_mol_data(
        prot, deprot, idx, selected_node_features, selected_edge_features,
    )
    loader = dataset_to_dataloader([m], 1, shuffle=False)
    assert np.isclose(predict_pka_value(query_model.model, loader), 5.32189131)

    # https://www.masterorganicchemistry.com/2017/04/18/basicity-of-amines-and-pkah/
    deprot = Chem.MolFromSmiles("C1CCNCC1")
    prot = Chem.MolFromSmiles("C1CC[NH2+]CC1")
    idx = get_ionization_indices([deprot, prot])[0]
    m = mol_to_paired_mol_data(
        prot, deprot, idx, selected_node_features, selected_edge_features,
    )
    loader = dataset_to_dataloader([m], 1, shuffle=False)
    assert np.isclose(predict_pka_value(query_model.model, loader), 11.02899933)


def test_piperidine():
    # https://www.masterorganicchemistry.com/2017/04/18/basicity-of-amines-and-pkah/
    mol = Chem.MolFromSmiles("C1CCNCC1")
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=True)
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )
    print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
    assert Chem.MolToSmiles(pair[0]) == "C1CC[NH2+]CC1"
    assert Chem.MolToSmiles(pair[1]) == "C1CCNCC1"
    assert np.isclose(pka, 11.02899933)
    assert idx == 3

    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    assert len(molpairs) == 1
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )
    print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
    assert Chem.MolToSmiles(pair[0]) == "C1CC[NH2+]CC1"
    assert Chem.MolToSmiles(pair[1]) == "C1CCNCC1"
    assert np.isclose(pka, 11.02899933)
    assert idx == 3


def test_pyridine():
    # https://www.masterorganicchemistry.com/2017/04/18/basicity-of-amines-and-pkah/
    mol = Chem.MolFromSmiles("C1=CC=NC=C1")
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=True)
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )
    print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
    assert Chem.MolToSmiles(pair[0]) == "c1cc[nH+]cc1"
    assert Chem.MolToSmiles(pair[1]) == "c1ccncc1"
    assert np.isclose(pka, 5.32189131)
    assert idx == 3


def test_acetic_acid():
    # https://en.wikipedia.org/wiki/Acetic_acid
    mol = Chem.MolFromSmiles("CC(=O)[O-]")
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )
    print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
    assert Chem.MolToSmiles(pair[0]) == "CC(=O)O"
    assert Chem.MolToSmiles(pair[1]) == "CC(=O)[O-]"
    assert np.isclose(pka, 4.81062174)
    assert idx == 3

    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=True)
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )
    print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
    assert Chem.MolToSmiles(pair[0]) == "CC(=O)O"
    assert Chem.MolToSmiles(pair[1]) == "CC(=O)[O-]"
    assert np.isclose(pka, 4.81062174)
    assert idx == 3


def test_fumaric_acid():
    # https://www.waters.com/nextgen/ca/en/library/application-notes/2020/analysis-of-organic-acids-using-a-mixed-mode-lc-column-and-an-acquity-qda-mass-detector.html
    mol = Chem.MolFromSmiles("O=C(O)/C=C/C(=O)O")
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(molpairs)):
        pka, pair, idx = (
            molpairs[i][0],
            molpairs[i][1],
            molpairs[i][2],
        )
        print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
        print(pka)
    print("################################")

    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "O=C(O)/C=C/C(=O)O"
    assert Chem.MolToSmiles(pair[1]) == "O=C([O-])/C=C/C(=O)O"
    assert np.isclose(pka, 3.52694106)
    assert idx == 2 or idx == 7
    ################################################
    protonation_state = 1
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )
    print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
    assert Chem.MolToSmiles(pair[0]) == "O=C([O-])/C=C/C(=O)O"
    assert Chem.MolToSmiles(pair[1]) == "O=C([O-])/C=C/C(=O)[O-]"
    assert np.isclose(pka, 4.94770622)
    assert idx == 2 or idx == 7

    ################################################
    ################################################
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=True)
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )
    print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
    assert Chem.MolToSmiles(pair[0]) == "O=C(O)/C=C/C(=O)O"
    assert Chem.MolToSmiles(pair[1]) == "O=C([O-])/C=C/C(=O)O"
    assert np.isclose(pka, 3.52694106)
    assert idx == 2
    ################################################
    protonation_state = 1
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )
    print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
    assert Chem.MolToSmiles(pair[0]) == "O=C([O-])/C=C/C(=O)O"
    assert Chem.MolToSmiles(pair[1]) == "O=C([O-])/C=C/C(=O)[O-]"
    assert np.isclose(pka, 4.94770622)
    assert idx == 7


def test_malic_acid():
    # https://www.waters.com/nextgen/ca/en/library/application-notes/2020/analysis-of-organic-acids-using-a-mixed-mode-lc-column-and-an-acquity-qda-mass-detector.html
    mol = Chem.MolFromSmiles("O=C(O)CC(O)C(=O)O")

    ################################################
    ################################################
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(molpairs)):
        pka, pair, idx = (
            molpairs[i][0],
            molpairs[i][1],
            molpairs[i][2],
        )
        print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
        print(pka)
    print("################################")

    assert len(molpairs) == 3
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )
    print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
    assert Chem.MolToSmiles(pair[0]) == "O=C(O)CC(O)C(=O)O"
    assert Chem.MolToSmiles(pair[1]) == "O=C(O)CC(O)C(=O)[O-]"
    assert np.isclose(pka, 2.79113841)
    assert idx == 8
    ################################################
    protonation_state = 1
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )
    print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
    assert Chem.MolToSmiles(pair[0]) == "O=C(O)CC(O)C(=O)[O-]"
    assert Chem.MolToSmiles(pair[1]) == "O=C([O-])CC(O)C(=O)[O-]"
    assert np.isclose(pka, 3.86118364)
    assert idx == 2
    ################################################
    protonation_state = 2
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "O=C([O-])CC(O)C(=O)[O-]"
    assert Chem.MolToSmiles(pair[1]) == "O=C([O-])CC([O-])C(=O)[O-]"
    assert np.isclose(pka, 12.24020576)
    assert idx == 5


def test_citric_acid():
    # https://www.waters.com/nextgen/ca/en/library/application-notes/2020/analysis-of-organic-acids-using-a-mixed-mode-lc-column-and-an-acquity-qda-mass-detector.html
    mol = Chem.MolFromSmiles("O=C(O)CC(O)(CC(=O)O)C(=O)O")

    ################################################
    ################################################
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    assert len(molpairs) == 4
    ################################################
    print("################################")
    for i in range(len(molpairs)):
        pka, pair, idx = (
            molpairs[i][0],
            molpairs[i][1],
            molpairs[i][2],
        )
        print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
        print(pka)
    print("################################")

    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )
    print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
    assert Chem.MolToSmiles(pair[0]) == "O=C(O)CC(O)(CC(=O)O)C(=O)O"
    assert Chem.MolToSmiles(pair[1]) == "O=C(O)CC(O)(CC(=O)O)C(=O)[O-]"
    assert np.isclose(pka, 2.97935152)
    assert idx == 12

    ################################################
    protonation_state = 1
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )
    print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
    assert Chem.MolToSmiles(pair[0]) == "O=C(O)CC(O)(CC(=O)O)C(=O)[O-]"
    assert Chem.MolToSmiles(pair[1]) == "O=C([O-])CC(O)(CC(=O)O)C(=O)[O-]"
    assert np.isclose(pka, 3.93864965)
    assert idx == 2 or idx == 9

    ################################################
    protonation_state = 2
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )
    print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
    assert Chem.MolToSmiles(pair[0]) == "O=C([O-])CC(O)(CC(=O)O)C(=O)[O-]"
    assert Chem.MolToSmiles(pair[1]) == "O=C([O-])CC(O)(CC(=O)[O-])C(=O)[O-]"
    assert np.isclose(pka, 4.28753757)
    assert idx == 9 or idx == 2
    ################################################
    protonation_state = 3
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "O=C([O-])CC(O)(CC(=O)[O-])C(=O)[O-]"
    assert Chem.MolToSmiles(pair[1]) == "O=C([O-])CC([O-])(CC(=O)[O-])C(=O)[O-]"
    assert np.isclose(pka, 12.02427483)
    assert idx == 5


def test_ascorbic_acid():
    # https://www.waters.com/nextgen/ca/en/library/application-notes/2020/analysis-of-organic-acids-using-a-mixed-mode-lc-column-and-an-acquity-qda-mass-detector.html
    mol = Chem.MolFromSmiles("C(C(C1C(=C(C(=O)O1)O)O)O)O")

    ################################################
    ################################################
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(molpairs)):
        pka, pair, idx = (
            molpairs[i][0],
            molpairs[i][1],
            molpairs[i][2],
        )
        print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
        print(pka)
    print("################################")
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )
    print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
    assert Chem.MolToSmiles(pair[0]) == "O=C1OC(C(O)CO)C(O)=C1O"
    assert Chem.MolToSmiles(pair[1]) == "O=C1OC(C(O)CO)C([O-])=C1O"
    assert np.isclose(pka, 3.39439988)
    assert idx == 9

    ################################################
    protonation_state = 1
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "O=C1OC(C(O)CO)C([O-])=C1O"
    assert Chem.MolToSmiles(pair[1]) == "O=C1OC(C(O)C[O-])C([O-])=C1O"
    assert np.isclose(pka, 9.08883286)
    assert idx == 7
    ################################################
    protonation_state = 2
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "O=C1OC(C(O)C[O-])C([O-])=C1O"
    assert Chem.MolToSmiles(pair[1]) == "O=C1OC(C([O-])C[O-])C([O-])=C1O"
    assert np.isclose(pka, 10.539691925048828)
    assert idx == 5


def test_quinic_acid():
    # https://www.waters.com/nextgen/ca/en/library/application-notes/2020/analysis-of-organic-acids-using-a-mixed-mode-lc-column-and-an-acquity-qda-mass-detector.html
    mol = Chem.MolFromSmiles("O=C(O)[C@]1(O)C[C@@H](O)[C@@H](O)[C@H](O)C1")

    ################################################
    ################################################
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(molpairs)):
        pka, pair, idx = (
            molpairs[i][0],
            molpairs[i][1],
            molpairs[i][2],
        )
        print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
        print(pka)
    print("################################")

    ################################################
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )
    print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
    assert Chem.MolToSmiles(pair[0]) == "O=C(O)[C@]1(O)C[C@@H](O)[C@@H](O)[C@H](O)C1"
    assert Chem.MolToSmiles(pair[1]) == "O=C([O-])[C@]1(O)C[C@@H](O)[C@@H](O)[C@H](O)C1"
    assert np.isclose(pka, 3.7188952)
    assert idx == 2
    ################################################
    protonation_state = 1
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "O=C([O-])[C@]1(O)C[C@@H](O)[C@@H](O)[C@H](O)C1"
    assert (
        Chem.MolToSmiles(pair[1]) == "O=C([O-])[C@]1([O-])C[C@@H](O)[C@@H](O)[C@H](O)C1"
    )
    assert np.isclose(pka, 12.78168488)
    assert idx == 4
    ################################################


def test_cocaine():
    # https://www.waters.com/nextgen/ca/en/library/application-notes/2020/analysis-of-organic-acids-using-a-mixed-mode-lc-column-and-an-acquity-qda-mass-detector.html
    mol = Chem.MolFromSmiles("COC(=O)C1C(OC(=O)C2=CC=CC=C2)CC2CCC1N2C")

    ################################################
    ################################################
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(molpairs)):
        pka, pair, idx = (
            molpairs[i][0],
            molpairs[i][1],
            molpairs[i][2],
        )
        print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
        print(pka)
    print("################################")

    assert len(molpairs) == 1
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "COC(=O)C1C(OC(=O)c2ccccc2)CC2CCC1[NH+]2C"
    assert Chem.MolToSmiles(pair[1]) == "COC(=O)C1C(OC(=O)c2ccccc2)CC2CCC1N2C"
    assert np.isclose(pka, 8.055152893066406)
    assert idx == 20
    ################################################


def test_tyrosine():
    # https://chem.libretexts.org/Courses/University_of_Arkansas_Little_Rock/CHEM_4320_5320%3A_Biochemistry_1/01%3A_Amino_Acids/1.4%3A_Reactions_of_Amino_Acids/1.4.1_Acid-base_Chemistry_of_Amino_Acids
    mol = Chem.MolFromSmiles("NC(CC1=CC=C(O)C=C1)C(=O)O")
    ################################################
    ################################################
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(molpairs)):
        pka, pair, idx = (
            molpairs[i][0],
            molpairs[i][1],
            molpairs[i][2],
        )
        print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
        print(pka)
    print("################################")
    ################################################
    assert len(molpairs) == 3
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "[NH3+]C(Cc1ccc(O)cc1)C(=O)O"
    assert Chem.MolToSmiles(pair[1]) == "[NH3+]C(Cc1ccc(O)cc1)C(=O)[O-]"
    assert np.isclose(pka, 2.2737984657287598)
    assert idx == 12
    ################################################
    protonation_state = 1
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "[NH3+]C(Cc1ccc(O)cc1)C(=O)[O-]"
    assert Chem.MolToSmiles(pair[1]) == "[NH3+]C(Cc1ccc([O-])cc1)C(=O)[O-]"
    assert np.isclose(pka, 8.993396759033203)
    assert idx == 7
    ################################################
    protonation_state = 2
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "[NH3+]C(Cc1ccc([O-])cc1)C(=O)[O-]"
    assert Chem.MolToSmiles(pair[1]) == "NC(Cc1ccc([O-])cc1)C(=O)[O-]"
    assert np.isclose(pka, 9.486579895019531)
    assert idx == 0
    ################################################


def test_aspartic_acid():
    # https://chem.libretexts.org/Courses/University_of_Arkansas_Little_Rock/CHEM_4320_5320%3A_Biochemistry_1/01%3A_Amino_Acids/1.4%3A_Reactions_of_Amino_Acids/1.4.1_Acid-base_Chemistry_of_Amino_Acids
    mol = Chem.MolFromSmiles("N[C@@H](CC(=O)O)C(=O)O")
    ################################################
    ################################################
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(molpairs)):
        pka, pair, idx = (
            molpairs[i][0],
            molpairs[i][1],
            molpairs[i][2],
        )
        print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
        print(pka)
    print("################################")
    ################################################
    assert len(molpairs) == 3
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "[NH3+][C@@H](CC(=O)O)C(=O)O"
    assert Chem.MolToSmiles(pair[1]) == "[NH3+][C@@H](CC(=O)O)C(=O)[O-]"
    assert np.isclose(pka, 2.21097993850708)
    assert idx == 8
    ################################################
    protonation_state = 1
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "[NH3+][C@@H](CC(=O)O)C(=O)[O-]"
    assert Chem.MolToSmiles(pair[1]) == "[NH3+][C@@H](CC(=O)[O-])C(=O)[O-]"
    assert np.isclose(pka, 3.519895076751709)
    assert idx == 5
    ################################################
    protonation_state = 2
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "[NH3+][C@@H](CC(=O)[O-])C(=O)[O-]"
    assert Chem.MolToSmiles(pair[1]) == "N[C@@H](CC(=O)[O-])C(=O)[O-]"
    assert np.isclose(pka, 9.840612411499023)
    assert idx == 0
    ################################################


def test_taurin():
    # http://www2.chm.ulaval.ca/gecha/chm1903/4_acide-base/organic_acids.pdf
    mol = Chem.MolFromSmiles("NCCS(=O)(=O)O")
    ################################################
    ################################################
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(molpairs)):
        pka, pair, idx = (
            molpairs[i][0],
            molpairs[i][1],
            molpairs[i][2],
        )
        print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
        print(pka)
    print("################################")
    ################################################
    assert len(molpairs) == 1
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "[NH3+]CCS(=O)(=O)[O-]"
    assert Chem.MolToSmiles(pair[1]) == "NCCS(=O)(=O)[O-]"
    assert np.isclose(pka, 9.672035217285156)
    assert idx == 0
    ################################################


def test_cysteamine():
    # http://www2.chm.ulaval.ca/gecha/chm1903/4_acide-base/organic_acids.pdf
    mol = Chem.MolFromSmiles("NCCS")
    ################################################
    ################################################
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(molpairs)):
        pka, pair, idx = (
            molpairs[i][0],
            molpairs[i][1],
            molpairs[i][2],
        )
        print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
        print(pka)
    print("################################")
    ################################################
    assert len(molpairs) == 3
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "[NH3+]CC[SH2+]"
    assert Chem.MolToSmiles(pair[1]) == "[NH3+]CCS"
    assert np.isclose(pka, 1.9965983629226685)
    assert idx == 3
    ################################################
    protonation_state = 1
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "[NH3+]CCS"
    assert Chem.MolToSmiles(pair[1]) == "[NH3+]CC[S-]"
    assert np.isclose(pka, 8.375046730041504)
    assert idx == 3
    ################################################
    protonation_state = 2
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "[NH3+]CC[S-]"
    assert Chem.MolToSmiles(pair[1]) == "NCC[S-]"
    assert np.isclose(pka, 10.587972640991211)
    assert idx == 0
    ################################################


def test_diphonoethane():
    # http://www2.chm.ulaval.ca/gecha/chm1903/4_acide-base/organic_acids.pdf
    mol = Chem.MolFromSmiles("CC(O)(P(=O)(O)O)P(=O)(O)O")
    ################################################
    ################################################
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(molpairs)):
        pka, pair, idx = (
            molpairs[i][0],
            molpairs[i][1],
            molpairs[i][2],
        )
        print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
        print(pka)
    print("################################")
    ################################################
    assert len(molpairs) == 4
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "CC(O)(P(=O)(O)O)P(=O)(O)O"
    assert Chem.MolToSmiles(pair[1]) == "CC(O)(P(=O)([O-])O)P(=O)(O)O"
    assert np.isclose(pka, 2.3572850227355957)
    assert idx == 9 or idx == 5  # symmetric mol
    ################################################
    protonation_state = 1
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "CC(O)(P(=O)([O-])O)P(=O)(O)O"
    assert Chem.MolToSmiles(pair[1]) == "CC(O)(P(=O)([O-])O)P(=O)([O-])O"
    assert np.isclose(pka, 2.4741861820220947)
    assert idx == 5 or idx == 9  # symmetric mol
    ################################################
    protonation_state = 2
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "CC(O)(P(=O)([O-])O)P(=O)([O-])O"
    assert Chem.MolToSmiles(pair[1]) == "CC(O)(P(=O)([O-])[O-])P(=O)([O-])O"
    assert np.isclose(pka, 5.930602550506592)
    assert idx == 6
    ################################################
    protonation_state = 3
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "CC(O)(P(=O)([O-])[O-])P(=O)([O-])O"
    assert Chem.MolToSmiles(pair[1]) == "CC(O)(P(=O)([O-])[O-])P(=O)([O-])[O-]"
    assert np.isclose(pka, 10.735816955566406)
    assert idx == 10
    ################################################


def test_arginin():
    # https://chem.libretexts.org/Courses/University_of_Arkansas_Little_Rock/CHEM_4320_5320%3A_Biochemistry_1/01%3A_Amino_Acids/1.4%3A_Reactions_of_Amino_Acids/1.4.1_Acid-base_Chemistry_of_Amino_Acids
    mol = Chem.MolFromSmiles("NC(N)=NCCCC(N)C(=O)O")
    ################################################
    ################################################
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(molpairs)):
        pka, pair, idx = (
            molpairs[i][0],
            molpairs[i][1],
            molpairs[i][2],
        )
        print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
        print(pka)
    print("################################")
    ################################################
    assert len(molpairs) == 3
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "NC(N)=[NH+]CCCC([NH3+])C(=O)O"
    assert Chem.MolToSmiles(pair[1]) == "NC(N)=[NH+]CCCC([NH3+])C(=O)[O-]"
    assert np.isclose(pka, 2.0857582092285156)
    assert idx == 11
    ################################################
    protonation_state = 1
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "NC(N)=[NH+]CCCC([NH3+])C(=O)[O-]"
    assert Chem.MolToSmiles(pair[1]) == "NC(N)=[NH+]CCCC(N)C(=O)[O-]"
    assert np.isclose(pka, 9.936403274536133)
    assert idx == 8
    ################################################
    protonation_state = 2
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "NC(N)=[NH+]CCCC(N)C(=O)[O-]"
    assert Chem.MolToSmiles(pair[1]) == "NC(N)=NCCCC(N)C(=O)[O-]"
    assert np.isclose(pka, 11.475444793701172)
    assert idx == 3
    ################################################


def test_thiophenecarboxylicacid():
    # http://www2.chm.ulaval.ca/gecha/chm1903/4_acide-base/organic_acids.pdf
    mol = Chem.MolFromSmiles("O=C(O)C1=CSC=C1")
    ################################################
    ################################################
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(molpairs)):
        pka, pair, idx = (
            molpairs[i][0],
            molpairs[i][1],
            molpairs[i][2],
        )
        print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
        print(pka)
    print("################################")
    ################################################
    assert len(molpairs) == 1
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "O=C(O)c1ccsc1"
    assert Chem.MolToSmiles(pair[1]) == "O=C([O-])c1ccsc1"
    assert np.isclose(pka, 3.9233970642089844)
    assert idx == 2  # symmetric mol


def test_nitroaniline():
    # http://www2.chm.ulaval.ca/gecha/chm1903/4_acide-base/organic_acids.pdf
    mol = Chem.MolFromSmiles("NC1=CC([N+](=O)[O-])=CC=C1")
    ################################################
    ################################################
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(molpairs)):
        pka, pair, idx = (
            molpairs[i][0],
            molpairs[i][1],
            molpairs[i][2],
        )
        print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
        print(pka)
    print("################################")
    ################################################
    assert len(molpairs) == 2
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "[NH3+]c1cccc([N+](=O)[O-])c1"
    assert Chem.MolToSmiles(pair[1]) == "Nc1cccc([N+](=O)[O-])c1"
    assert np.isclose(pka, 2.331765651702881)
    assert idx == 0
    ################################################
    protonation_state = 1
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "Nc1cccc([N+](=O)[O-])c1"
    assert Chem.MolToSmiles(pair[1]) == "[NH-]c1cccc([N+](=O)[O-])c1"
    assert np.isclose(pka, 11.329208374023438)
    assert idx == 0


def test_benzenesulfinicacid():
    # http://www2.chm.ulaval.ca/gecha/chm1903/4_acide-base/organic_acids.pdf
    mol = Chem.MolFromSmiles("O=S(O)C1=CC=CC=C1")
    ################################################
    ################################################
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(molpairs)):
        pka, pair, idx = (
            molpairs[i][0],
            molpairs[i][1],
            molpairs[i][2],
        )
        print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
        print(pka)
    print("################################")
    ################################################
    assert len(molpairs) == 1
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "O=S(O)c1ccccc1"
    assert Chem.MolToSmiles(pair[1]) == "O=S([O-])c1ccccc1"
    assert np.isclose(pka, 2.209087371826172)
    assert idx == 2


def test_bromobenzoicacid():
    # http://www2.chm.ulaval.ca/gecha/chm1903/4_acide-base/organic_acids.pdf
    mol = Chem.MolFromSmiles("O=C(O)C1=CC(Br)=CC=C1")
    ################################################
    ################################################
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(molpairs)):
        pka, pair, idx = (
            molpairs[i][0],
            molpairs[i][1],
            molpairs[i][2],
        )
        print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
        print(pka)
    print("################################")
    ################################################
    assert len(molpairs) == 1
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert Chem.MolToSmiles(pair[0]) == "O=C(O)c1cccc(Br)c1"
    assert Chem.MolToSmiles(pair[1]) == "O=C([O-])c1cccc(Br)c1"
    assert np.isclose(pka, 3.643803596496582)
    assert idx == 2


def test_benzaldehyde():
    # http://www2.chm.ulaval.ca/gecha/chm1903/4_acide-base/organic_acids.pdf
    mol = Chem.MolFromSmiles("O=CC1=CC=CC=C1")
    ################################################
    ################################################
    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    print("################################")
    for i in range(len(molpairs)):
        pka, pair, idx = (
            molpairs[i][0],
            molpairs[i][1],
            molpairs[i][2],
        )
        print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
        print(pka)
    print("################################")
    ################################################
    assert len(molpairs) == 0
    ################################################



def test_mol_00():
    # 00 Chembl molecule
    mol = mollist[0]

    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )
    print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
    assert (
        Chem.MolToSmiles(pair[0])
        == "[NH3+]c1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc([NH3+])c5ccccc54)c3)c2)c2ccccc12"
    )
    assert (
        Chem.MolToSmiles(pair[1])
        == "Nc1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc([NH3+])c5ccccc54)c3)c2)c2ccccc12"
    )
    assert np.isclose(pka, 1.861354112625122)
    assert idx == 21 or idx == 0
    ################################################
    protonation_state = 1
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert (
        Chem.MolToSmiles(pair[0])
        == "Nc1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc([NH3+])c5ccccc54)c3)c2)c2ccccc12"
    )
    assert (
        Chem.MolToSmiles(pair[1])
        == "Nc1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc(N)c5ccccc54)c3)c2)c2ccccc12"
    )
    assert np.isclose(pka, 2.4510035514831543)
    assert idx == 0 or idx == 21

    ################################################
    protonation_state = 2
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )
    assert (
        Chem.MolToSmiles(pair[0])
        == "Nc1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc(N)c5ccccc54)c3)c2)c2ccccc12"
    )
    assert (
        Chem.MolToSmiles(pair[1])
        == "[NH-]c1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc(N)c5ccccc54)c3)c2)c2ccccc12"
    )

    assert np.isclose(pka, 11.28736496)
    assert idx == 0 or idx == 21
    ################################################
    protonation_state = 3
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert (
        Chem.MolToSmiles(pair[0])
        == "[NH-]c1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc(N)c5ccccc54)c3)c2)c2ccccc12"
    )
    assert (
        Chem.MolToSmiles(pair[1])
        == "[NH-]c1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc([NH-])c5ccccc54)c3)c2)c2ccccc12"
    )

    assert np.isclose(pka, 11.68689919)
    assert idx == 21 or idx == 0

    print("#####################################################")
    print("#####################################################")

    molpairs = calculate_microstate_pka_values(mol, only_dimorphite=True)
    ################################################
    protonation_state = 0
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )

    assert (
        Chem.MolToSmiles(pair[0])
        == "[NH3+]c1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc([NH3+])c5ccccc54)c3)c2)c2ccccc12"
    )
    assert (
        Chem.MolToSmiles(pair[1])
        == "Nc1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc([NH3+])c5ccccc54)c3)c2)c2ccccc12"
    )
    assert np.isclose(pka, 1.74858117)
    assert idx == 0 or idx == 21
    ################################################
    protonation_state = 1
    pka, pair, idx = (
        molpairs[protonation_state][0],
        molpairs[protonation_state][1],
        molpairs[protonation_state][2],
    )
    assert (
        Chem.MolToSmiles(pair[0])
        == "Nc1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc([NH3+])c5ccccc54)c3)c2)c2ccccc12"
    )
    assert (
        Chem.MolToSmiles(pair[1])
        == "Nc1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc(N)c5ccccc54)c3)c2)c2ccccc12"
    )

    assert np.isclose(pka, 2.39621401)
    assert idx == 21 or idx == 0


# def test_mol_14():
#     # 14th Chembl molecule
#     mol = mollist[14]
#     molpairs = calculate_microstate_pka_values(mol, only_dimorphite=False)
#     ################################################
#     protonation_state = 0
#     pka, pair, idx = (
#         molpairs[protonation_state][0],
#         molpairs[protonation_state][1],
#         molpairs[protonation_state][2],
#     )
#     print("###################################")
#     print(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
#     print(pka)
#     assert Chem.MolToSmiles(pair[0]) == "O[NH2+]C1c2cc(O)c(O)cc2-c2cc(O)c(O)cc21"
#     assert Chem.MolToSmiles(pair[1]) == "ONC1c2cc(O)c(O)cc2-c2cc(O)c(O)cc21"

#     assert Chem.MolToSmiles(molpairs[1][0]) == "ONC1c2cc(O)c(O)cc2-c2cc(O)c(O)cc21"
#     assert Chem.MolToSmiles(molpairs[1][1]) == "[O-]c1cc2c(cc1O)-c1cc(O)c(O)cc1C2NO"

#     assert Chem.MolToSmiles(molpairs[2][0]) == "[O-]c1cc2c(cc1O)-c1cc(O)c(O)cc1C2NO"
#     assert Chem.MolToSmiles(molpairs[2][1]) == "[O-]c1[c-]c2c(cc1O)-c1cc(O)c(O)cc1C2NO"

#     assert Chem.MolToSmiles(molpairs[3][0]) == "[O-]c1[c-]c2c(cc1O)-c1cc(O)c(O)cc1C2NO"
#     assert (
#         Chem.MolToSmiles(molpairs[3][1]) == "[O-]c1[c-]c2c(cc1[O-])-c1cc(O)c(O)cc1C2NO"
#     )

#     assert (
#         Chem.MolToSmiles(molpairs[4][0]) == "[O-]c1[c-]c2c(cc1[O-])-c1cc(O)c(O)cc1C2NO"
#     )
#     assert (
#         Chem.MolToSmiles(molpairs[4][1])
#         == "[O-]c1[c-]c2c(cc1[O-])-c1[c-]c(O)c(O)cc1C2NO"
#     )

#     assert (
#         Chem.MolToSmiles(molpairs[5][0])
#         == "[O-]c1[c-]c2c(cc1[O-])-c1[c-]c(O)c(O)cc1C2NO"
#     )
#     assert (
#         Chem.MolToSmiles(molpairs[5][1])
#         == "[O-]c1[c-]c2c(cc1[O-])-c1[c-]c(O)c(O)cc1C2[N-]O"
#     )

#     assert (
#         Chem.MolToSmiles(molpairs[6][0])
#         == "[O-]c1[c-]c2c(cc1[O-])-c1[c-]c(O)c(O)cc1C2[N-]O"
#     )
#     assert (
#         Chem.MolToSmiles(molpairs[6][1])
#         == "[O-][N-]C1c2[c-]c([O-])c([O-])cc2-c2[c-]c(O)c(O)cc21"
#     )

#     assert pkas == [5.004, 9.134, 10.041, 10.558, 10.958, 11.986, 12.938]
#     assert atoms == [1, 16, 17, 14, 9, 1, 0]


# def test_mol_20():
#     # 20th Chembl molecule
#     mol = Chem.MolToSmiles(mollist[20])
#     molpairs, pkas, atoms = smiles_query(mol, output_smiles=True)
#     print(molpairs, pkas, atoms)
#     assert (
#         molpairs[0][0]
#         == "CC1(C)C2=CCC3C4CC[C@H](OC5CC5)[C@@]4(C)CCC3[C@@]2(C)CC[C@@H]1O"
#     )
#     assert (
#         molpairs[0][1]
#         == "CC1(C)C2=CCC3C4CC[C@H](OC5CC5)[C@@]4(C)CCC3[C@@]2(C)CC[C@@H]1[O-]"
#     )

#     assert pkas == [11.336]
#     assert atoms == [25]


# def test_mol_42():
#     # 42th Chembl molecule
#     mol = Chem.MolToInchi(mollist[42])
#     molpairs, pkas, atoms = inchi_query(mol, output_inchi=True)

#     assert (
#         molpairs[0][0]
#         == "InChI=1S/C16H15NS2/c1-3-12-8-14-15(18-12)9-17-10-16(14)19-13-6-4-11(2)5-7-13/h4-10H,3H2,1-2H3/p+1"
#     )
#     assert (
#         molpairs[0][1]
#         == "InChI=1S/C16H15NS2/c1-3-12-8-14-15(18-12)9-17-10-16(14)19-13-6-4-11(2)5-7-13/h4-10H,3H2,1-2H3"
#     )

#     assert pkas == [4.369]
#     assert atoms == [15]


# def test_mol_47():
#     # 47th Chembl molecule
#     mol = Chem.MolToInchi(mollist[47])
#     molpairs, pkas, atoms = inchi_query(mol, output_inchi=True)

#     assert (
#         molpairs[0][0]
#         == "InChI=1S/C18H20F2O2/c1-3-13(11-5-7-17(21)15(19)9-11)14(4-2)12-6-8-18(22)16(20)10-12/h5-10,13-14,21-22H,3-4H2,1-2H3"
#     )
#     assert (
#         molpairs[0][1]
#         == "InChI=1S/C18H20F2O2/c1-3-13(11-5-7-17(21)15(19)9-11)14(4-2)12-6-8-18(22)16(20)10-12/h5-10,13-14,21-22H,3-4H2,1-2H3/p-1"
#     )
#     assert (
#         molpairs[1][0]
#         == "InChI=1S/C18H20F2O2/c1-3-13(11-5-7-17(21)15(19)9-11)14(4-2)12-6-8-18(22)16(20)10-12/h5-10,13-14,21-22H,3-4H2,1-2H3/p-1"
#     )
#     assert (
#         molpairs[1][1]
#         == "InChI=1S/C18H20F2O2/c1-3-13(11-5-7-17(21)15(19)9-11)14(4-2)12-6-8-18(22)16(20)10-12/h5-10,13-14,21-22H,3-4H2,1-2H3/p-2"
#     )
#     assert pkas == [8.243, 9.255]
#     assert atoms == [7, 18]


# def test_mol_53():
#     # 53th Chembl molecule
#     mol = Chem.MolToInchi(mollist[53])
#     molpairs, pkas, atoms = inchi_query(mol, output_inchi=True)

#     assert (
#         molpairs[0][0]
#         == "InChI=1S/C29H44O8/c1-15-26(33)21(30)12-24(36-15)37-18-6-8-27(2)17(11-18)4-5-20-19(27)7-9-28(3)25(16-10-23(32)35-14-16)22(31)13-29(20,28)34/h10,15,17-22,24-26,30-31,33-34H,4-9,11-14H2,1-3H3/t15?,17?,18?,19?,20?,21?,22?,24?,25-,26?,27?,28?,29?/m0/s1"
#     )
#     assert (
#         molpairs[0][1]
#         == "InChI=1S/C29H43O8/c1-15-26(33)21(30)12-24(36-15)37-18-6-8-27(2)17(11-18)4-5-20-19(27)7-9-28(3)25(16-10-23(32)35-14-16)22(31)13-29(20,28)34/h10,15,17-22,24-26,30-31,33H,4-9,11-14H2,1-3H3/q-1/t15?,17?,18?,19?,20?,21?,22?,24?,25-,26?,27?,28?,29?/m0/s1"
#     )
#     assert (
#         molpairs[1][0]
#         == "InChI=1S/C29H43O8/c1-15-26(33)21(30)12-24(36-15)37-18-6-8-27(2)17(11-18)4-5-20-19(27)7-9-28(3)25(16-10-23(32)35-14-16)22(31)13-29(20,28)34/h10,15,17-22,24-26,30-31,33H,4-9,11-14H2,1-3H3/q-1/t15?,17?,18?,19?,20?,21?,22?,24?,25-,26?,27?,28?,29?/m0/s1"
#     )
#     assert (
#         molpairs[1][1]
#         == "InChI=1S/C29H42O8/c1-15-26(33)21(30)12-24(36-15)37-18-6-8-27(2)17(11-18)4-5-20-19(27)7-9-28(3)25(16-10-23(32)35-14-16)22(31)13-29(20,28)34/h10,15,17-22,24-26,30,33H,4-9,11-14H2,1-3H3/q-2/t15?,17?,18?,19?,20?,21?,22?,24?,25-,26?,27?,28?,29?/m0/s1"
#     )

#     assert (
#         molpairs[2][0]
#         == "InChI=1S/C29H42O8/c1-15-26(33)21(30)12-24(36-15)37-18-6-8-27(2)17(11-18)4-5-20-19(27)7-9-28(3)25(16-10-23(32)35-14-16)22(31)13-29(20,28)34/h10,15,17-22,24-26,30,33H,4-9,11-14H2,1-3H3/q-2/t15?,17?,18?,19?,20?,21?,22?,24?,25-,26?,27?,28?,29?/m0/s1"
#     )
#     assert (
#         molpairs[2][1]
#         == "InChI=1S/C29H41O8/c1-15-26(33)21(30)12-24(36-15)37-18-6-8-27(2)17(11-18)4-5-20-19(27)7-9-28(3)25(16-10-23(32)35-14-16)22(31)13-29(20,28)34/h10,15,17-22,24-26,30H,4-9,11-14H2,1-3H3/q-3/t15?,17?,18?,19?,20?,21?,22?,24?,25-,26?,27?,28?,29?/m0/s1"
#     )

#     assert (
#         molpairs[3][0]
#         == "InChI=1S/C29H41O8/c1-15-26(33)21(30)12-24(36-15)37-18-6-8-27(2)17(11-18)4-5-20-19(27)7-9-28(3)25(16-10-23(32)35-14-16)22(31)13-29(20,28)34/h10,15,17-22,24-26,30H,4-9,11-14H2,1-3H3/q-3/t15?,17?,18?,19?,20?,21?,22?,24?,25-,26?,27?,28?,29?/m0/s1"
#     )
#     assert (
#         molpairs[3][1]
#         == "InChI=1S/C29H40O8/c1-15-26(33)21(30)12-24(36-15)37-18-6-8-27(2)17(11-18)4-5-20-19(27)7-9-28(3)25(16-10-23(32)35-14-16)22(31)13-29(20,28)34/h10,15,17-22,24-26H,4-9,11-14H2,1-3H3/q-4/t15?,17?,18?,19?,20?,21?,22?,24?,25-,26?,27?,28?,29?/m0/s1"
#     )

#     assert pkas == [10.081, 12.85, 13.321, 13.357]
#     assert atoms == [30, 27, 36, 34]


# def test_mol_58():
#     # 58th Chembl molecule
#     mol = Chem.MolToSmiles(mollist[58])
#     molpairs, pkas, atoms = smiles_query(mol, output_smiles=True)

#     assert molpairs[0][0] == "CCCCCC[NH+]1CC[NH+]2CC(c3ccccc3)c3ccccc3C2C1"
#     assert molpairs[0][1] == "CCCCCCN1CC[NH+]2CC(c3ccccc3)c3ccccc3C2C1"

#     assert molpairs[1][0] == "CCCCCCN1CC[NH+]2CC(c3ccccc3)c3ccccc3C2C1"
#     assert molpairs[1][1] == "CCCCCCN1CCN2CC(c3ccccc3)c3ccccc3C2C1"

#     assert pkas == [5.578, 8.163]
#     assert atoms == [6, 9]


# def test_mol_59():
#     # 59th Chembl molecule
#     mol = Chem.MolToSmiles(mollist[59])
#     molpairs, pkas, atoms = smiles_query(mol, output_smiles=True)

#     assert molpairs[0][0] == "CC/C(=C(\c1ccc(I)cc1)c1ccc(OCC[NH+](C)C)cc1)c1ccccc1"
#     assert molpairs[0][1] == "CC/C(=C(\c1ccc(I)cc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"

#     assert pkas == [8.484]
#     assert atoms == [18]


# def test_mol_62():
#     # 62th Chembl molecule
#     mol = Chem.MolToSmiles(mollist[62])
#     molpairs, pkas, atoms = smiles_query(mol, output_smiles=True)

#     assert molpairs[0][0] == "Cc1cc(CCCCCOc2ccc(-c3[nH+]c(C)c(C)o3)cc2)o[nH+]1"
#     assert molpairs[0][1] == "Cc1cc(CCCCCOc2ccc(-c3nc(C)c(C)o3)cc2)o[nH+]1"

#     assert molpairs[1][0] == "Cc1cc(CCCCCOc2ccc(-c3nc(C)c(C)o3)cc2)o[nH+]1"
#     assert molpairs[1][1] == "Cc1cc(CCCCCOc2ccc(-c3nc(C)c(C)o3)cc2)on1"

#     assert pkas == [1.477, 2.326]
#     assert atoms == [15, 24]


# def test_mol_70():
#     # 70th Chembl molecule
#     mol = Chem.MolToSmiles(mollist[70])
#     molpairs, pkas, atoms = smiles_query(mol, output_smiles=True)

#     assert molpairs[0][0] == "Oc1ccc(/C(=C(/c2ccc(O)cc2)C(F)(F)F)C(F)(F)F)cc1"
#     assert molpairs[0][1] == "[O-]c1ccc(/C(=C(/c2ccc(O)cc2)C(F)(F)F)C(F)(F)F)cc1"

#     assert molpairs[1][0] == "[O-]c1ccc(/C(=C(/c2ccc(O)cc2)C(F)(F)F)C(F)(F)F)cc1"
#     assert molpairs[1][1] == "[O-]c1ccc(/C(=C(/c2ccc([O-])cc2)C(F)(F)F)C(F)(F)F)cc1"

#     assert pkas == [7.991, 9.039]
#     assert atoms == [0, 11]
