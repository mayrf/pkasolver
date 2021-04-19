from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

def morgan_fp(df, mol_column, neigh: int, nB: int, useFeatures=True):
    """Take Pandas DataFrame, creates Morgan fingerprints from the molecules in "mol_column"
    and add them to DataFrame.
    """
    df[mol_column + "_morganfp"] = [
        GetMorganFingerprintAsBitVect(
            m, neigh, nBits=nB, useFeatures=useFeatures
        )
        for m in df[mol_column]
    ]
    return df

def create_conjugate(mol, id, pka, pH=7.4):
    """Create a new molecule that is the conjugated base/acid to the input molecule."""
    mol = Chem.RWMol(mol)
    atom = mol.GetAtomWithIdx(id)
    charge = atom.GetFormalCharge()
    Ex_Hs = atom.GetNumExplicitHs()
    Tot_Hs = atom.GetTotalNumHs()

    #
    if pka > pH and Tot_Hs > 0:
        atom.SetFormalCharge(charge - 1)
        if Ex_Hs > 0:
            atom.SetNumExplicitHs(Ex_Hs - 1)
    elif pka < pH:
        atom.SetFormalCharge(charge + 1)
        if Tot_Hs == 0 or Ex_Hs > 0:
            atom.SetNumExplicitHs(Ex_Hs + 1)
    else:
        # pka > pH and Tot_Hs < 0
        atom.SetFormalCharge(charge + 1)
        if Tot_Hs == 0 or Ex_Hs > 0:
            atom.SetNumExplicitHs(Ex_Hs + 1)

    atom.UpdatePropertyCache()
    return mol
