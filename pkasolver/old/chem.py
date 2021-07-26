from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import numpy as np

def create_conjugate(mol, id, pka, pH=7.4):
    """Create a new molecule that is the conjugated base/acid to the input molecule."""
    mol = Chem.RWMol(mol)
    atom = mol.GetAtomWithIdx(id)
    charge = atom.GetFormalCharge()
    Ex_Hs = atom.GetNumExplicitHs()
    Tot_Hs = atom.GetTotalNumHs()

#with correction for +2 charges            
    if  (pka > pH and Tot_Hs > 0) or charge > 0:
        atom.SetFormalCharge(charge - 1)
        if Ex_Hs > 0:
            atom.SetNumExplicitHs(Ex_Hs - 1)
            
    elif pka < pH and charge <= 0:
        atom.SetFormalCharge(charge + 1)
        if Tot_Hs == 0 or Ex_Hs > 0:
            atom.SetNumExplicitHs(Ex_Hs + 1)
    
    else:
        # pka > pH and Tot_Hs = 0
        atom.SetFormalCharge(charge + 1)
        if Tot_Hs == 0 or Ex_Hs > 0:
            atom.SetNumExplicitHs(Ex_Hs + 1)

    atom.UpdatePropertyCache()
    return mol

def morgan_fp_array(df, mol_column, nBits=4096, radius=3, useFeatures=True):
    """Take Pandas DataFrame, create Morgan fingerprints from the molecules in "mol_column"
    and return them as numpy array.
    """
    length = len(df)
    i = 0
    for mol in df[mol_column]:
        
        fp = np.array(GetMorganFingerprintAsBitVect(
            mol, radius=3, nBits=4096, useFeatures=useFeatures
        ))
        if i == 0:
            array = fp
        else:
            array = np.vstack((array, fp))
        if i%1000 == 0:
            print(f'fp:{i} of {length}')
        i += 1
    return array

#return whether a bond matches a particular smarts query
def bond_smarts_query(bond, smarts):
    for match in bond.GetOwningMol().GetSubstructMatches(Chem.MolFromSmarts(smarts)):
        if set((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())) == set(match):
            return True
    return False

def atom_smarts_query(atom, smarts):
    if atom.GetOwningMol().GetSubstructMatches(Chem.MolFromSmarts(smarts)):
        if atom.GetIdx() in atom.GetOwningMol().GetSubstructMatches(Chem.MolFromSmarts(smarts))[0]:
                return True
    return False

def tanimoto(fp1, fp2):
    set1= set(fp1.nonzero()[0].tolist())
    set2= set(fp2.nonzero()[0].tolist())
    return len(set1&set2)/len(set1|set2)




# def morgan_fp(df, mol_column, neigh: int, nB: int, useFeatures=True):
#     """Take Pandas DataFrame, creates Morgan fingerprints from the molecules in "mol_column"
#     and add them to DataFrame.
#     """
#     df[mol_column + "_morganfp"] = [
#         GetMorganFingerprintAsBitVect(
#             m, neigh, nBits=nB, useFeatures=useFeatures
#         )
#         for m in df[mol_column]
#     ]
#     return df


# from rdkit.Chem import PandasTools, AllChem as Chem, Descriptors
# from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
# # In many cases NaN
# # not_used_desc = ['MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge']
# not_used_desc = []

# # Create a descriptor calculator for all RDKit descriptors except the ones above
# desc_calc = MolecularDescriptorCalculator([x for x in [x[0] for x in Descriptors.descList] if x not in not_used_desc])
# # len(Descriptors.descList)

# def mol_desc_array(df, mol_column):
#     """Take Pandas DataFrame, create Morgan fingerprints from the molecules in "mol_column"
#     and return them as numpy array.
#     """
#     length = len(df)
#     for i, mol in enumerate(df[mol_column]):
#         row = desc_calc.CalcDescriptors(mol)
        
#         if i == 0:
#             array = row
#         else:
#             array = np.vstack((array, row))
#         if i%1000 == 0:
#             print(f'desc:{i} of {length}')
#         i += 1
#     return array