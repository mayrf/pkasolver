import logging

from pkasolver.chem import (atom_smarts_query, bond_smarts_query,
                            make_smarts_features)

logger = logging.getLogger(__name__)

import torch

NUM_THREADS = 1
torch.set_num_threads(NUM_THREADS)
logger.debug(f"Setting num threads to {NUM_THREADS}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
logger.debug(f"Pytorch will use {DEVICE}")

# Defining Smarts patterns used to calculate some node and edge features
rotatable_bond = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
rotatable_bond_no_amide = "[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]"  # any good? https://rdkit-discuss.narkive.com/4o99LqS6/rotatable-bonds-amide-bonds-and-smarts
amide = "[NX3][CX3](=[OX1])[#6]"
keton = "[CX3]=[OX1]"

# from https://molvs.readthedocs.io/en/latest/_modules/molvs/charge.html
smarts_dict = {
    "-OSO3H": ["OS(=O)(=O)[OH]", "OS(=O)(=O)[O-]"],  #
    "–SO3H": ["[!O]S(=O)(=O)[OH]", "[!O]S(=O)(=O)[O-]"],  #
    "-OSO2H": ["O[SD3](=O)[OH]", "O[SD3](=O)[O-]"],
    "-SO2H": ["[!O][SD3](=O)[OH]", "[!O][SD3](=O)[O-]"],
    "-OPO3H2": ["OP(=O)([OH])[OH]", "OP(=O)([OH])[O-]"],
    "-PO3H2": ["[!O]P(=O)([OH])[OH]", "[!O]P(=O)([OH])[O-]"],
    "-CO2H": ["C(=O)[OH]", "C(=O)[O-]"],
    "thiophenol": ["c[SH]", "c[S-]"],
    "(-OPO3H)-": ["OP(=O)([O-])[OH]", "OP(=O)([O-])[O-]"],
    "(-PO3H)-": ["[!O]P(=O)([O-])[OH]", "[!O]P(=O)([O-])[O-]"],
    "phthalimide": ["O=C2c1ccccc1C(=O)[NH]2", "O=C2c1ccccc1C(=O)[N-]2"],
    "CO3H (peracetyl)": ["C(=O)O[OH]", "C(=O)O[O-]"],
    "alpha-carbon-hydrogen-nitro group": ["O=N(O)[CH]", "O=N(O)[C-]"],
    "-SO2NH2": ["S(=O)(=O)[NH2]", "S(=O)(=O)[NH-]"],
    "-OBO2H2": ["OB([OH])[OH]", "OB([OH])[O-]"],
    "-BO2H2": ["[!O]B([OH])[OH]", "[!O]B([OH])[O-]"],
    "phenol": ["c[OH]", "c[O-]"],
    "SH (aliphatic)": ["C[SH]", "C[S-]"],
    "(-OBO2H)-": ["OB([O-])[OH]", "OB([O-])[O-]"],
    "(-BO2H)-": ["[!O]B([O-])[OH]", "[!O]B([O-])[O-]"],
    "cyclopentadiene": ["C1=CC=C[CH2]1", "c1ccc[cH-]1"],
    "-CONH2": ["C(=O)[NH2]", "C(=O)[NH-]"],
    "imidazole": ["c1cnc[nH]1" "c1cnc[n-]1"],
    "-OH (aliphatic alcohol)": ["[CX4][OH]", "[CX4][O-]"],
    "alpha-carbon-hydrogen-keto group": ["O=C([!O])[C!H0+0]", "O=C([!O])[C-]"],
    "alpha-carbon-hydrogen-acetyl ester group": ["OC(=O)[C!H0+0]", "OC(=O)[C-]"],
    "sp carbon hydrogen": ["C#[CH]", "C#[C-]"],
    "alpha-carbon-hydrogen-sulfone group": ["CS(=O)(=O)[C!H0+0]", "CS(=O)(=O)[C-]"],
    "alpha-carbon-hydrogen-sulfoxide group": ["C[SD3](=O)[C!H0+0]", "C[SD3](=O)[C-]"],
    "-NH2": ["[CX4][NH2]", "[CX4][NH-]"],
    "benzyl hydrogen": ["c[CX4H2]", "c[CX3H-]"],
    "sp2-carbon hydrogen": ["[CX3]=[CX3!H0+0]", "[CX3]=[CX2-]"],
    "sp3-carbon hydrogen": ["[CX4!H0+0]", "[CX3-]"],  #
    "Hydrogen-bond acceptor": [  #
        "[#6,#7;R0]=[#8]",
        "[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]",
    ],
    "Hydrogen-bond donor": ["[!$([#6,H0,-,-2,-3])]", "[!H0;#7,#8,#9]"],
    "Possible intramolecular H-bond": ["[O,N;!H0]-*~*-*=[$([C,N;R0]=O)]"],
}

node_feat_values = {
    "element": [
        1,
        6,
        7,
        8,
        9,
        15,
        16,
        17,
        33,
        35,
        53,
    ],  # still missing to mark element that's not in the list
    "formal_charge": [-1, 0, 1],
    "is_in_ring": [1],
    "amide_center_atom": [1],
    "hybridization": [1, 2, 3, 4],
    "total_num_Hs": [0, 1, 2, 3],
    "aromatic_tag": [1],
    "total_valence": [1, 2, 3, 4, 5, 6],
    "total_degree": [1, 2, 3, 4],
    "reaction_center": [1],
    "smarts": smarts_dict.keys(),
}

# defining helper dictionaries for generating one hot encoding of atom features
NODE_FEATURES = {
    "element": lambda atom, marvin_atom: list(
        map(lambda s: int(atom.GetAtomicNum() == s), node_feat_values["element"],)
    ),  # still missing to mark element that's not in the list
    "formal_charge": lambda atom, marvin_atom: list(
        map(
            lambda s: int(atom.GetFormalCharge() == s),
            node_feat_values["formal_charge"],
        )
    ),
    "is_in_ring": lambda atom, marvin_atom: atom.IsInRing(),
    "amide_center_atom": lambda atom, marvin_atom: atom_smarts_query(atom, amide),
    "hybridization": lambda atom, marvin_atom: list(
        map(
            lambda s: int(atom.GetHybridization() == s),
            node_feat_values["hybridization"],
        )
    ),
    "total_num_Hs": lambda atom, marvin_atom: list(
        map(lambda s: int(atom.GetTotalNumHs() == s), node_feat_values["total_num_Hs"],)
    ),
    "aromatic_tag": lambda atom, marvin_atom: atom.GetIsAromatic(),
    "total_valence": lambda atom, marvin_atom: list(
        map(
            lambda s: int(atom.GetTotalValence() == s),
            node_feat_values["total_valence"],
        )
    ),
    "total_degree": lambda atom, marvin_atom: list(
        map(
            lambda s: int(atom.GetTotalDegree() == s), node_feat_values["total_degree"],
        )
    ),
    "reaction_center": lambda atom, marvin_atom: atom.GetIdx() == int(marvin_atom),
    "smarts": lambda atom, marvin_atom: make_smarts_features(atom, smarts_dict),
}

# defining possible edge feature values
edge_feat_values = {
    "bond_type": [1.0, 1.5, 2.0, 3.0],
    "is_conjugated": [1],
    "rotatable": [1],
}

# defining helper dictionaries for generating one hot encoding of edge features
EDGE_FEATURES = {
    "bond_type": lambda bond: list(
        map(
            lambda s: int(bond.GetBondTypeAsDouble() == s),
            edge_feat_values["bond_type"],
        )
    ),
    "is_conjugated": lambda bond: bond.GetIsConjugated(),
    "rotatable": lambda bond: bond_smarts_query(bond, rotatable_bond),
}
