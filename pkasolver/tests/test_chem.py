from pkasolver.chem import create_conjugate
from pkasolver.data import load_data
from rdkit import Chem


def test_conjugates():
    from pkasolver.data import import_sdf

    sdf_filepaths = load_data()
    df = import_sdf(sdf_filepaths["Training"])

    for i in range(len(df.index)):
        mol_unchanged = Chem.Mol(df.ROMol[i])
        index = int(df.marvin_atom[i])
        pka = float(df.marvin_pKa[i])
        mol_new = create_conjugate(mol_unchanged, index, pka, ignore_danger=True)
        print(
            mol_unchanged.GetNumHeavyAtoms()
            - mol_unchanged.GetNumAtoms(onlyExplicit=False),
            mol_new.GetNumHeavyAtoms() - mol_new.GetNumAtoms(onlyExplicit=False),
        )
        assert Chem.MolToSmiles(mol_unchanged) != Chem.MolToSmiles(mol_new)


def test_correctness_of_conjugates_for_bases():

    m = Chem.MolFromSmiles(
        "[NH3+]c1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc([NH3+])c5ccccc54)c3)c2)c2ccccc12"
    )
    mol_new = create_conjugate(m, 21, 10.5, ignore_danger=True)
    print(Chem.MolToSmiles(mol_new))
    assert (
        Chem.MolToSmiles(mol_new)
        == "Nc1cc[n+](Cc2cccc(-c3cccc(C[n+]4ccc([NH3+])c5ccccc54)c3)c2)c2ccccc12"
    )


def test_correctness_of_conjugates_for_acids():

    m = Chem.MolFromSmiles("CC(=O)[O-]")
    mol_new = create_conjugate(m, 3, 2.5, ignore_danger=True)
    print(Chem.MolToSmiles(mol_new))
    Chem.MolToSmiles(mol_new) == "CC(=O)O"
