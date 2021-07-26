from pkasolver import chem

rotatable_bond = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
rotatable_bond_no_amide = '[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]' #any good? https://rdkit-discuss.narkive.com/4o99LqS6/rotatable-bonds-amide-bonds-and-smarts
amide = '[NX3][CX3](=[OX1])[#6]'
keton = '[CX3]=[OX1]'

NODE_FEATURES = {
    'atomic_number':lambda atom, i, marvin_atom: atom.GetAtomicNum(),
    'formal_charge': lambda atom, i, marvin_atom:atom.GetFormalCharge(),
    'chiral_tag': lambda atom, i, marvin_atom:atom.GetChiralTag(),
    'is_in_ring': lambda atom, i, marvin_atom:atom.IsInRing(),
    'amide_center_atom': lambda atom, i, marvin_atom: chem.atom_smarts_query(atom, amide),
    'hybridization': lambda atom, i, marvin_atom: atom.GetHybridization(),
    'total_num_Hs': lambda atom, i, marvin_atom: atom.GetTotalNumHs(),
    'explicit_num_Hs': lambda atom, i, marvin_atom: atom.GetNumExplicitHs(),
    'aromatic_tag': lambda atom, i, marvin_atom: atom.GetIsAromatic(),
    'total_valence': lambda atom, i, marvin_atom: atom.GetTotalValence(),
    'total_degree': lambda atom, i, marvin_atom: atom.GetTotalDegree(),
    'marvin_atom': lambda atom, i, marvin_atom: i == int(marvin_atom)
}

EDGE_FEATURES = {
    'bond_type':lambda bond: bond.GetBondTypeAsDouble(), 
    'is_conjugated': lambda bond: bond.GetIsConjugated(),
    'rotatable': lambda bond: chem.bond_smarts_query(bond, rotatable_bond)
}