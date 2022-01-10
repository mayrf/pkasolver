import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import ResonanceMolSupplier
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from copy import deepcopy


def create_conjugate(
    mol_initial: Chem.Mol,
    idx: int,
    pka: float,
    pH: float = 7.4,
    ignore_danger: bool = False,
):
    """Create a new molecule that is the conjugated base/acid to the input molecule."""
    mol = deepcopy(mol_initial)
    mol_changed = Chem.RWMol(mol)
    Chem.SanitizeMol(mol_changed)
    atom = mol_changed.GetAtomWithIdx(idx)
    charge = atom.GetFormalCharge()
    Ex_Hs = atom.GetNumExplicitHs()
    Tot_Hs = atom.GetTotalNumHs()
    danger = False
    # make deprotonated conjugate as pKa > pH with at least one proton or
    # mol charge is positive (otherwise conjugate reaction center would have charge +2 --> highly unlikely)
    if (pka > pH and Tot_Hs > 0):
        atom.SetFormalCharge(charge - 1)
        if Ex_Hs > 0:
            atom.SetNumExplicitHs(Ex_Hs - 1)

    # make protonated conjugate as pKa < pH and charge is neutral or negative
    elif pka <= pH and charge <= 0:
        atom.SetFormalCharge(charge + 1)
        if Tot_Hs == 0 or Ex_Hs > 0:
            atom.SetNumExplicitHs(Ex_Hs + 1)

    # make protonated conjugate as pKa > pH and there are no proton at the reaction center
    elif pka > pH and Tot_Hs == 0:
        atom.SetFormalCharge(charge + 1)
        danger = True
        if Tot_Hs == 0 or Ex_Hs > 0:
            atom.SetNumExplicitHs(Ex_Hs + 1)

    else:
        raise RuntimeError(
            f"pka: {pka},charge:{charge},Explicit Hs:{Ex_Hs}, Total Hs:{Tot_Hs}, reaction center atomic number: {atom.GetAtomicNum()}"
        )
    atom.UpdatePropertyCache()

    if atom.GetSymbol() == "O" and atom.GetFormalCharge() == 1:
        raise RuntimeError("Protonating already protonated oxygen. Aborting.")

    Tot_Hs_after = atom.GetTotalNumHs()
    assert Tot_Hs != Tot_Hs_after
    # mol = next(ResonanceMolSupplier(mol))
    if danger and not ignore_danger:
        print(f"Original mol: {Chem.MolToSmiles(mol)}")
        print(f"Changed mol: {Chem.MolToSmiles(mol_changed)}")
        print(
            f"This should only happen for the test set. pka: {pka},charge:{charge},Explicit Hs:{Ex_Hs}, Total Hs:{Tot_Hs}, reaction center atomic number: {atom.GetAtomicNum()}"
        )
        raise RuntimeError("danger")
    return mol_changed


# def generate_morgan_fp_array(df, mol_column, nBits=4096, radius=3, useFeatures=True):
#     """Take Pandas DataFrame, create Morgan fingerprints from the molecules in "mol_column"
#     and return them as numpy array.
#     """
#     length = len(df)
#     i = 0
#     for mol in df[mol_column]:

#         fp = np.array(
#             GetMorganFingerprintAsBitVect(
#                 mol, radius=3, nBits=4096, useFeatures=useFeatures
#             )
#         )
#         if i == 0:
#             array = fp
#         else:
#             array = np.vstack((array, fp))
#         if i % 1000 == 0:
#             print(f"fp:{i} of {length}")
#         i += 1
#     return array


def bond_smarts_query(bond, smarts):
    # return whether a bond matches a particular smarts query

    for match in bond.GetOwningMol().GetSubstructMatches(Chem.MolFromSmarts(smarts)):
        if set((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) == set(match):
            return True
    return False


def atom_smarts_query(atom, smarts):
    return atom.GetIdx() in sum(
        atom.GetOwningMol().GetSubstructMatches(Chem.MolFromSmarts(smarts)), ()
    )


def make_smarts_features(atom, smarts_dict):
    bits = []
    for lst in smarts_dict.values():
        i = 0
        for smarts in lst:
            if atom_smarts_query(atom, smarts):
                i = 1
                continue
        bits.append(i)
    return bits


# def calculate_tanimoto_coefficient(fp1, fp2):
#     set1 = set(fp1.nonzero()[0].tolist())
#     set2 = set(fp2.nonzero()[0].tolist())
#     return len(set1 & set2) / len(set1 | set2)
