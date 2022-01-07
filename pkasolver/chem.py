from copy import deepcopy

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.rdchem import ResonanceMolSupplier


def create_conjugate(
    mol_initial: Chem.rdchem.Mol,
    atom_idx: int,
    pka: float,
    pH: float = 7.4,
    ignore_danger: bool = False,
) -> Chem.rdchem.Mol:
    """Create the conjugated base/acid of the input molecule

    whether the input molecule is the protonated or
    deprotonated molecule respective to the acid-base reaction
    in question is inferred from the pka and pH input.
    If the resulting molecule is illegal, e.g. has negative number of protons,
    or highly unlikely, e.g. atom charge of +2 or -2, the opposite ionization state is output instead

    Parameters
    ----------
    mol_initial
        molecule object in the protonation state at the specified pH
    atom_idx
        atom index of ionization center of acid-base reaction
    pka
        pka value of the acid-base reaction in question
    pH
        pH of the ionization state of the input molecule
    ignore_danger
        If false, runtime error is raised if conjugate molecule is illegal or highly unlikely.
        If true, opposite conjugate is output, without raising runtime error

    Raises
    ------
    RuntimeError
        is raised if conjugate molecule is illegal or highly unlikely and ignore_danger is set to False

    Returns
    -------
    Chem.Mol
        conjugate molecule

    """
    mol = deepcopy(mol_initial)
    mol_changed = Chem.RWMol(mol)
    Chem.SanitizeMol(mol_changed)
    atom = mol_changed.GetAtomWithIdx(atom_idx)
    charge = atom.GetFormalCharge()
    Ex_Hs = atom.GetNumExplicitHs()
    Tot_Hs = atom.GetTotalNumHs()
    danger = False
    # make deprotonated conjugate as pKa > pH with at least one proton or
    # mol charge is positive (otherwise conjugate reaction center would have charge +2 --> highly unlikely)
    if (pka > pH and Tot_Hs > 0) or charge > 0:
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
    Tot_Hs_after = atom.GetTotalNumHs()
    assert Tot_Hs != Tot_Hs_after
    if danger and not ignore_danger:
        print(f"Original mol: {Chem.MolToSmiles(mol)}")
        print(f"Changed mol: {Chem.MolToSmiles(mol_changed)}")
        print(
            f"This should only happen for the test set. pka: {pka},charge:{charge},Explicit Hs:{Ex_Hs}, Total Hs:{Tot_Hs}, reaction center atomic number: {atom.GetAtomicNum()}"
        )
        raise RuntimeError("danger")
    return mol_changed


def generate_morgan_fp_array(
    mol_lst: list,
    nBits: int = 4096,
    radius: int = 3,
    useFeatures: bool = True,
) -> np.ndarray:
    """Takes a list of molecules, creates Morgan fingerprints
    and returns them as numpy array.

    Parameters
    ----------
    mol_lst
        list of Chem.Mol object
    nBits
        number of morgan fingerprint bits
    radius
        fingerprint connectivity radius
    useFeatures
        if True, include molecules features for Morgan fp generation


    Returns
    -------
    np.ndarray
        array with molecules as rows and fingerprint bits as columns

    """
    for i, mol in enumerate(mol_lst):
        fp = np.array(
            GetMorganFingerprintAsBitVect(
                mol, radius=radius, nBits=nBits, useFeatures=useFeatures
            )
        )
        if i == 0:
            array = fp
        else:
            array = np.vstack((array, fp))
        if i % 1000 == 0:
            print(f"fp:{i} of {len(mol_lst)}")

    return array


# return whether a bond matches a particular smarts query
def bond_smarts_query(bond: Chem.rdchem.Bond, smarts: str) -> bool:
    """Checks if bond is part of a substructure that matches the smarts pattern.

    Parameters
    ----------
    bond
        bond to be matched with smarts
    smarts
        smarts to be matched

    Returns
    -------
    bool
        returns True if bond is part of a substructure that matches the smarts pattern
    """
    for match in bond.GetOwningMol().GetSubstructMatches(Chem.MolFromSmarts(smarts)):
        if set((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) == set(match):
            return True
    return False


def atom_smarts_query(atom: Chem.rdchem.Atom, smarts: str) -> bool:
    """Checks if atom is part of a substructure that matches the smarts pattern.

    Parameters
    ----------
    atom
        atom to be matched with smarts
    smarts
        smarts to be matched

    Returns
    -------
    bool
        returns True if atom is part of a substructure that matches the smarts pattern
    """
    return atom.GetIdx() in sum(
        atom.GetOwningMol().GetSubstructMatches(Chem.MolFromSmarts(smarts)), ()
    )


def make_smarts_features(atom: Chem.rdchem.Atom, smarts_dict: dict) -> list:
    """Returns list of bits, one for each smarts string in smarts_dict,
    indicating whether the input atom is part of a substructure matching the smarts pattern.

    Parameters
    ----------
    atom
        atom to be matched with smarts pattern
    smarts_dict
        dict of smarts strings

    Returns
    -------
    list
        list of bits indicating smarts pattern matching results
    """
    bits = []
    for lst in smarts_dict.values():
        i = 0
        for smarts in lst:
            if atom_smarts_query(atom, smarts):
                i = 1
                continue
        bits.append(i)
    return bits


def calculate_tanimoto_coefficient(fp1: list, fp2: list) -> float:
    """Calculate tanimoto coefficient of the two input fingerprint lists

    Parameters
    ----------
    fp1, fp2
        list of fingerprint bits to be matched against each other

    Returns
    -------
    float
        tanimoto coefficient of input fingerprints
    """
    set1 = set(fp1.nonzero()[0].tolist())
    set2 = set(fp2.nonzero()[0].tolist())
    return len(set1 & set2) / len(set1 | set2)
