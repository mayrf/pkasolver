import logging
from copy import deepcopy

from rdkit import Chem

logger = logging.getLogger(__name__)


def create_conjugate(
    mol_initial: Chem.rdchem.Mol,
    idx: int,
    pka: float,
    pH: float = 7.4,
    ignore_danger: bool = False,
    known_pka_values: bool = True,
) -> Chem.rdchem.Mol:

    """Create the conjugated base/acid of the input molecule depending on if the input molecule is the protonated or
    deprotonated molecule in the acid-base reaction. This is inferred from the pka and pH input.
    If the resulting molecule is illegal, e.g. has negative number of protons on a heavy atom, or highly unlikely, 
    e.g. atom charge of +2 or -2, the opposite ionization state is returned instead
    
    Parameters
    ----------
    mol_initial
        molecule from which either a proton is removed or added
    atom_idx
        atom index of ionization center of the acid-base reaction
    pka
        pka value of the acid-base reaction
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
    Chem.rdchem.Mol
        conjugated molecule
    """

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
    if (pka > pH and Tot_Hs > 0) or (charge > 0 and known_pka_values):
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

    if (
        atom.GetSymbol() == "O"
        and atom.GetFormalCharge() == 1
        and known_pka_values == False
    ):
        raise RuntimeError("Protonating already protonated oxygen. Aborting.")

    Tot_Hs_after = atom.GetTotalNumHs()
    assert Tot_Hs != Tot_Hs_after
    # mol = next(ResonanceMolSupplier(mol))
    if danger and not ignore_danger:
        logger.debug(f"Original mol: {Chem.MolToSmiles(mol)}")
        logger.debug(f"Changed mol: {Chem.MolToSmiles(mol_changed)}")
        logger.debug(
            f"This should only happen for the test set. pka: {pka},charge:{charge},Explicit Hs:{Ex_Hs}, Total Hs:{Tot_Hs}, reaction center atomic number: {atom.GetAtomicNum()}"
        )
        raise RuntimeError("danger")
    return mol_changed


def bond_smarts_query(bond, smarts):
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


def atom_smarts_query(atom, smarts) -> bool:
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


def make_smarts_features(atom, smarts_dict):
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
