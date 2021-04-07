NODE_FEATURES = {
    'atomic_number':lambda atom, i, marvin_atom: atom.GetAtomicNum(),
    'formal_charge': lambda atom, i, marvin_atom:atom.GetFormalCharge(),
    'chiral_tag': lambda atom, i, marvin_atom:atom.GetChiralTag(),
    'hybridization': lambda atom, i, marvin_atom: atom.GetHybridization(),
    'explicit_Hs_number': lambda atom, i, marvin_atom: atom.GetNumExplicitHs(),
    'aromatic_tag': lambda atom, i, marvin_atom: atom.GetIsAromatic(),
    'total_valence': lambda atom, i, marvin_atom: atom.GetTotalValence(),
    'total_degree': lambda atom, i, marvin_atom: atom.GetTotalDegree(),
    'marvin_atom': lambda atom, i, marvin_atom: i == int(marvin_atom)
}

EDGE_FEATURES = {
    'bond_type':lambda bond: bond.GetBondTypeAsDouble(), 
    'is_conjugated': lambda bond: bond.GetIsConjugated()
}