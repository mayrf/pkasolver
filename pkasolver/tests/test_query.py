from rdkit import Chem
from pkasolver import query
import numpy as np

input = "pkasolver/tests/testdata/00_chembl_subset.sdf"
mollist = []
with open(input, "rb") as fh:
    suppl = Chem.ForwardSDMolSupplier(fh, removeHs=True)
    for i, mol in enumerate(suppl):
        mollist.append(mol)
        # if i >= 9:
        #     break


def test_mol_00():
    # 00 Chembl molecule
    mol = mollist[0]
    molpairs, pkas, atoms = query.mol_query(mollist[0])
    assert (
        Chem.MolToSmiles(molpairs[0][0])
        == "[NH3+]c1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc([NH3+])c5ccccc54)c3)c2)c2ccccc12"
    )
    assert (
        Chem.MolToSmiles(molpairs[0][1])
        == "Nc1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc([NH3+])c5ccccc54)c3)c2)c2ccccc12"
    )
    assert (
        Chem.MolToSmiles(molpairs[1][0])
        == "Nc1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc([NH3+])c5ccccc54)c3)c2)c2ccccc12"
    )
    assert (
        Chem.MolToSmiles(molpairs[1][1])
        == "Nc1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc(N)c5ccccc54)c3)c2)c2ccccc12"
    )

    assert (
        Chem.MolToSmiles(molpairs[2][0])
        == "Nc1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc(N)c5ccccc54)c3)c2)c2ccccc12"
    )
    assert (
        Chem.MolToSmiles(molpairs[2][1])
        == "[NH-]c1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc(N)c5ccccc54)c3)c2)c2ccccc12"
    )
    assert (
        Chem.MolToSmiles(molpairs[3][0])
        == "[NH-]c1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc(N)c5ccccc54)c3)c2)c2ccccc12"
    )
    assert (
        Chem.MolToSmiles(molpairs[3][1])
        == "[NH-]c1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc([NH-])c5ccccc54)c3)c2)c2ccccc12"
    )

    assert pkas == [1.893, 2.451, 11.287, 11.687]
    assert atoms == [21, 0, 0, 21]


def test_mol_14():
    # 14th Chembl molecule
    mol = Chem.MolToSmiles(mollist[14])
    molpairs, pkas, atoms = query.smiles_query(mol, output_smiles=True)

    assert molpairs[0][0] == "O[NH2+]C1c2cc(O)c(O)cc2-c2cc(O)c(O)cc21"
    assert molpairs[0][1] == "ONC1c2cc(O)c(O)cc2-c2cc(O)c(O)cc21"

    assert molpairs[1][0] == "ONC1c2cc(O)c(O)cc2-c2cc(O)c(O)cc21"
    assert molpairs[1][1] == "[O-]c1cc2c(cc1O)-c1cc(O)c(O)cc1C2NO"

    assert molpairs[2][0] == "[O-]c1cc2c(cc1O)-c1cc(O)c(O)cc1C2NO"
    assert molpairs[2][1] == "[O-]c1[c-]c2c(cc1O)-c1cc(O)c(O)cc1C2NO"

    assert molpairs[3][0] == "[O-]c1[c-]c2c(cc1O)-c1cc(O)c(O)cc1C2NO"
    assert molpairs[3][1] == "[O-]c1[c-]c2c(cc1[O-])-c1cc(O)c(O)cc1C2NO"

    assert molpairs[4][0] == "[O-]c1[c-]c2c(cc1[O-])-c1cc(O)c(O)cc1C2NO"
    assert molpairs[4][1] == "[O-]c1[c-]c2c(cc1[O-])-c1[c-]c(O)c(O)cc1C2NO"

    assert molpairs[5][0] == "[O-]c1[c-]c2c(cc1[O-])-c1[c-]c(O)c(O)cc1C2NO"
    assert molpairs[5][1] == "[O-]c1[c-]c2c(cc1[O-])-c1[c-]c(O)c(O)cc1C2[N-]O"

    assert molpairs[6][0] == "[O-]c1[c-]c2c(cc1[O-])-c1[c-]c(O)c(O)cc1C2[N-]O"
    assert molpairs[6][1] == "[O-][N-]C1c2[c-]c([O-])c([O-])cc2-c2[c-]c(O)c(O)cc21"

    assert pkas == [5.004, 9.134, 10.041, 10.558, 10.958, 11.986, 12.938]
    assert atoms == [1, 16, 17, 14, 9, 1, 0]


def test_mol_20():
    # 20th Chembl molecule
    mol = Chem.MolToSmiles(mollist[20])
    molpairs, pkas, atoms = query.smiles_query(mol, output_smiles=True)

    assert (
        molpairs[0][0]
        == "CC1(C)C2=CCC3C4CC[C@H](OC5CC5)[C@@]4(C)CCC3[C@@]2(C)CC[C@@H]1O"
    )
    assert (
        molpairs[0][1]
        == "CC1(C)C2=CCC3C4CC[C@H](OC5CC5)[C@@]4(C)CCC3[C@@]2(C)CC[C@@H]1[O-]"
    )

    assert pkas == [11.336]
    assert atoms == [25]


def test_mol_42():
    # 42th Chembl molecule
    mol = Chem.MolToInchi(mollist[42])
    molpairs, pkas, atoms = query.inchi_query(mol, output_inchi=True)

    assert (
        molpairs[0][0]
        == "InChI=1S/C16H15NS2/c1-3-12-8-14-15(18-12)9-17-10-16(14)19-13-6-4-11(2)5-7-13/h4-10H,3H2,1-2H3/p+1"
    )
    assert (
        molpairs[0][1]
        == "InChI=1S/C16H15NS2/c1-3-12-8-14-15(18-12)9-17-10-16(14)19-13-6-4-11(2)5-7-13/h4-10H,3H2,1-2H3"
    )

    assert pkas == [4.369]
    assert atoms == [15]


def test_mol_47():
    # 47th Chembl molecule
    mol = Chem.MolToInchi(mollist[47])
    molpairs, pkas, atoms = query.inchi_query(mol, output_inchi=True)

    assert (
        molpairs[0][0]
        == "InChI=1S/C18H20F2O2/c1-3-13(11-5-7-17(21)15(19)9-11)14(4-2)12-6-8-18(22)16(20)10-12/h5-10,13-14,21-22H,3-4H2,1-2H3"
    )
    assert (
        molpairs[0][1]
        == "InChI=1S/C18H20F2O2/c1-3-13(11-5-7-17(21)15(19)9-11)14(4-2)12-6-8-18(22)16(20)10-12/h5-10,13-14,21-22H,3-4H2,1-2H3/p-1"
    )
    assert (
        molpairs[1][0]
        == "InChI=1S/C18H20F2O2/c1-3-13(11-5-7-17(21)15(19)9-11)14(4-2)12-6-8-18(22)16(20)10-12/h5-10,13-14,21-22H,3-4H2,1-2H3/p-1"
    )
    assert (
        molpairs[1][1]
        == "InChI=1S/C18H20F2O2/c1-3-13(11-5-7-17(21)15(19)9-11)14(4-2)12-6-8-18(22)16(20)10-12/h5-10,13-14,21-22H,3-4H2,1-2H3/p-2"
    )
    assert pkas == [8.243, 9.255]
    assert atoms == [7, 18]


def test_mol_53():
    # 53th Chembl molecule
    mol = Chem.MolToInchi(mollist[53])
    molpairs, pkas, atoms = query.inchi_query(mol, output_inchi=True)

    assert (
        molpairs[0][0]
        == "InChI=1S/C29H44O8/c1-15-26(33)21(30)12-24(36-15)37-18-6-8-27(2)17(11-18)4-5-20-19(27)7-9-28(3)25(16-10-23(32)35-14-16)22(31)13-29(20,28)34/h10,15,17-22,24-26,30-31,33-34H,4-9,11-14H2,1-3H3/t15?,17?,18?,19?,20?,21?,22?,24?,25-,26?,27?,28?,29?/m0/s1"
    )
    assert (
        molpairs[0][1]
        == "InChI=1S/C29H43O8/c1-15-26(33)21(30)12-24(36-15)37-18-6-8-27(2)17(11-18)4-5-20-19(27)7-9-28(3)25(16-10-23(32)35-14-16)22(31)13-29(20,28)34/h10,15,17-22,24-26,30-31,33H,4-9,11-14H2,1-3H3/q-1/t15?,17?,18?,19?,20?,21?,22?,24?,25-,26?,27?,28?,29?/m0/s1"
    )
    assert (
        molpairs[1][0]
        == "InChI=1S/C29H43O8/c1-15-26(33)21(30)12-24(36-15)37-18-6-8-27(2)17(11-18)4-5-20-19(27)7-9-28(3)25(16-10-23(32)35-14-16)22(31)13-29(20,28)34/h10,15,17-22,24-26,30-31,33H,4-9,11-14H2,1-3H3/q-1/t15?,17?,18?,19?,20?,21?,22?,24?,25-,26?,27?,28?,29?/m0/s1"
    )
    assert (
        molpairs[1][1]
        == "InChI=1S/C29H42O8/c1-15-26(33)21(30)12-24(36-15)37-18-6-8-27(2)17(11-18)4-5-20-19(27)7-9-28(3)25(16-10-23(32)35-14-16)22(31)13-29(20,28)34/h10,15,17-22,24-26,30,33H,4-9,11-14H2,1-3H3/q-2/t15?,17?,18?,19?,20?,21?,22?,24?,25-,26?,27?,28?,29?/m0/s1"
    )

    assert (
        molpairs[2][0]
        == "InChI=1S/C29H42O8/c1-15-26(33)21(30)12-24(36-15)37-18-6-8-27(2)17(11-18)4-5-20-19(27)7-9-28(3)25(16-10-23(32)35-14-16)22(31)13-29(20,28)34/h10,15,17-22,24-26,30,33H,4-9,11-14H2,1-3H3/q-2/t15?,17?,18?,19?,20?,21?,22?,24?,25-,26?,27?,28?,29?/m0/s1"
    )
    assert (
        molpairs[2][1]
        == "InChI=1S/C29H41O8/c1-15-26(33)21(30)12-24(36-15)37-18-6-8-27(2)17(11-18)4-5-20-19(27)7-9-28(3)25(16-10-23(32)35-14-16)22(31)13-29(20,28)34/h10,15,17-22,24-26,30H,4-9,11-14H2,1-3H3/q-3/t15?,17?,18?,19?,20?,21?,22?,24?,25-,26?,27?,28?,29?/m0/s1"
    )

    assert (
        molpairs[3][0]
        == "InChI=1S/C29H41O8/c1-15-26(33)21(30)12-24(36-15)37-18-6-8-27(2)17(11-18)4-5-20-19(27)7-9-28(3)25(16-10-23(32)35-14-16)22(31)13-29(20,28)34/h10,15,17-22,24-26,30H,4-9,11-14H2,1-3H3/q-3/t15?,17?,18?,19?,20?,21?,22?,24?,25-,26?,27?,28?,29?/m0/s1"
    )
    assert (
        molpairs[3][1]
        == "InChI=1S/C29H40O8/c1-15-26(33)21(30)12-24(36-15)37-18-6-8-27(2)17(11-18)4-5-20-19(27)7-9-28(3)25(16-10-23(32)35-14-16)22(31)13-29(20,28)34/h10,15,17-22,24-26H,4-9,11-14H2,1-3H3/q-4/t15?,17?,18?,19?,20?,21?,22?,24?,25-,26?,27?,28?,29?/m0/s1"
    )

    assert pkas == [10.081, 12.85, 13.321, 13.357]
    assert atoms == [30, 27, 36, 34]


def test_mol_58():
    # 58th Chembl molecule
    mol = Chem.MolToSmiles(mollist[58])
    molpairs, pkas, atoms = query.smiles_query(mol, output_smiles=True)

    assert molpairs[0][0] == "CCCCCC[NH+]1CC[NH+]2CC(c3ccccc3)c3ccccc3C2C1"
    assert molpairs[0][1] == "CCCCCCN1CC[NH+]2CC(c3ccccc3)c3ccccc3C2C1"

    assert molpairs[1][0] == "CCCCCCN1CC[NH+]2CC(c3ccccc3)c3ccccc3C2C1"
    assert molpairs[1][1] == "CCCCCCN1CCN2CC(c3ccccc3)c3ccccc3C2C1"

    assert pkas == [5.578, 8.163]
    assert atoms == [6, 9]


def test_mol_59():
    # 59th Chembl molecule
    mol = Chem.MolToSmiles(mollist[59])
    molpairs, pkas, atoms = query.smiles_query(mol, output_smiles=True)

    assert molpairs[0][0] == "CC/C(=C(\c1ccc(I)cc1)c1ccc(OCC[NH+](C)C)cc1)c1ccccc1"
    assert molpairs[0][1] == "CC/C(=C(\c1ccc(I)cc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"

    assert pkas == [8.484]
    assert atoms == [18]


def test_mol_62():
    # 62th Chembl molecule
    mol = Chem.MolToSmiles(mollist[62])
    molpairs, pkas, atoms = query.smiles_query(mol, output_smiles=True)

    assert molpairs[0][0] == "Cc1cc(CCCCCOc2ccc(-c3[nH+]c(C)c(C)o3)cc2)o[nH+]1"
    assert molpairs[0][1] == "Cc1cc(CCCCCOc2ccc(-c3nc(C)c(C)o3)cc2)o[nH+]1"

    assert molpairs[1][0] == "Cc1cc(CCCCCOc2ccc(-c3nc(C)c(C)o3)cc2)o[nH+]1"
    assert molpairs[1][1] == "Cc1cc(CCCCCOc2ccc(-c3nc(C)c(C)o3)cc2)on1"

    assert pkas == [1.477, 2.326]
    assert atoms == [15, 24]


def test_mol_70():
    # 70th Chembl molecule
    mol = Chem.MolToSmiles(mollist[70])
    molpairs, pkas, atoms = query.smiles_query(mol, output_smiles=True)

    assert molpairs[0][0] == "Oc1ccc(/C(=C(/c2ccc(O)cc2)C(F)(F)F)C(F)(F)F)cc1"
    assert molpairs[0][1] == "[O-]c1ccc(/C(=C(/c2ccc(O)cc2)C(F)(F)F)C(F)(F)F)cc1"

    assert molpairs[1][0] == "[O-]c1ccc(/C(=C(/c2ccc(O)cc2)C(F)(F)F)C(F)(F)F)cc1"
    assert molpairs[1][1] == "[O-]c1ccc(/C(=C(/c2ccc([O-])cc2)C(F)(F)F)C(F)(F)F)cc1"

    assert pkas == [7.991, 9.039]
    assert atoms == [0, 11]
