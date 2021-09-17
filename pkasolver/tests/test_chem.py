from pkasolver.chem import create_conjugate
import numpy as np
from .test_data_generation import load_data
from rdkit import Chem


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