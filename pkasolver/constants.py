from pkasolver import chem
import torch

NUM_THREADS = 1
torch.set_num_threads(NUM_THREADS)
print(f"Setting num threads to {NUM_THREADS}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
print(f"Pytorch will use {DEVICE}")

rotatable_bond = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
rotatable_bond_no_amide = "[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]"  # any good? https://rdkit-discuss.narkive.com/4o99LqS6/rotatable-bonds-amide-bonds-and-smarts
amide = "[NX3][CX3](=[OX1])[#6]"
keton = "[CX3]=[OX1]"


# NODE_FEATURES = {
#     "element_onehot": lambda atom, i, marvin_atom: list(
#         map(
#             lambda s: float(atom.GetAtomicNum() == s),
#             [1, 6, 7, 8, 9, 15, 16, 17, 33, 35, 53],
#         )
#     ),  # still missing to mark atom that's not in the list
#     "formal_charge": lambda atom, i, marvin_atom: list(
#         map(lambda s: float(atom.GetFormalCharge() == s), [-1, 0, 1])
#     ),
#     "is_in_ring": lambda atom, i, marvin_atom: atom.IsInRing(),
#     "amide_center_atom": lambda atom, i, marvin_atom: chem.atom_smarts_query(
#         atom, amide
#     ),
#     "hybridization": lambda atom, i, marvin_atom: list(
#         map(lambda s: float(atom.GetHybridization() == s), [1, 2, 3, 4])
#     ),
#     "total_num_Hs": lambda atom, i, marvin_atom: list(
#         map(lambda s: float(atom.GetTotalNumHs() == s), [0, 1, 2, 3])
#     ),
#     "explicit_num_Hs": lambda atom, i, marvin_atom: atom.GetNumExplicitHs(),
#     "aromatic_tag": lambda atom, i, marvin_atom: atom.GetIsAromatic(),
#     "total_valence": lambda atom, i, marvin_atom: lambda atom, i, marvin_atom: list(
#         map(lambda s: float(atom.GetTotalValence() == s), [0, 1, 2, 3, 4, 5, 6])
#     ),
#     "total_degree": lambda atom, i, marvin_atom: list(
#         map(lambda s: float(atom.GetTotalDegree() == s), [1, 2, 3, 4])
#     ),
#     "reaction_center": lambda atom, i, marvin_atom: i == int(marvin_atom),
# }

NODE_FEATURES = {
    "element_onehot": lambda atom, i, marvin_atom: list(
        map(
            lambda s: float(atom.GetAtomicNum() == s),
            [1, 6, 7, 8, 9, 15, 16, 17, 33, 35, 53],
        )
    ),  # still missing to mark atom that's not in the list
    "formal_charge": lambda atom, i, marvin_atom: list(
        map(lambda s: float(atom.GetFormalCharge() == s), [-1, 0, 1])
    ),
    "is_in_ring": lambda atom, i, marvin_atom: atom.IsInRing(),
    "amide_center_atom": lambda atom, i, marvin_atom: chem.atom_smarts_query(
        atom, amide
    ),
    # "hybridization": lambda atom, i, marvin_atom: atom.GetHybridization(),
    "hybridization": lambda atom, i, marvin_atom: list(
        map(lambda s: float(atom.GetHybridization() == s), [1, 2, 3, 4])
    ),
    "total_num_Hs": lambda atom, i, marvin_atom: list(
        map(lambda s: float(atom.GetTotalNumHs() == s), [0, 1, 2, 3])
    ),
    "aromatic_tag": lambda atom, i, marvin_atom: atom.GetIsAromatic(),
    "total_valence": lambda atom, i, marvin_atom: list(
        map(lambda s: float(atom.GetTotalValence() == s), [1, 2, 3, 4, 5, 6])
    ),
    "total_degree": lambda atom, i, marvin_atom: list(
        map(lambda s: float(atom.GetTotalDegree() == s), [1, 2, 3, 4])
    ),
    "reaction_center": lambda atom, i, marvin_atom: i == int(marvin_atom),
}

# NODE_FEATURES = {
#     "atomic_number": lambda atom, i, marvin_atom: atom.GetAtomicNum(),
#     "element_onehot": lambda atom, i, marvin_atom: list(
#         map(
#             lambda s: float(atom.GetAtomicNum() == s),
#             [1, 6, 7, 8, 9, 15, 16, 17, 33, 35, 53],
#         )
#     ),  # still missing to mark atom that's not in the list
#     "formal_charge": lambda atom, i, marvin_atom: atom.GetFormalCharge(),
#     "chiral_tag": lambda atom, i, marvin_atom: atom.GetChiralTag(),
#     "is_in_ring": lambda atom, i, marvin_atom: atom.IsInRing(),
#     "amide_center_atom": lambda atom, i, marvin_atom: chem.atom_smarts_query(
#         atom, amide
#     ),
#     # "hybridization": lambda atom, i, marvin_atom: atom.GetHybridization(),
#     "hybridization": lambda atom, i, marvin_atom: list(
#         map(lambda s: float(atom.GetHybridization() == s), [1, 2, 3, 4])
#     ),
#     "total_num_Hs": lambda atom, i, marvin_atom: atom.GetTotalNumHs(),
#     "explicit_num_Hs": lambda atom, i, marvin_atom: atom.GetNumExplicitHs(),
#     "aromatic_tag": lambda atom, i, marvin_atom: atom.GetIsAromatic(),
#     "total_valence": lambda atom, i, marvin_atom: atom.GetTotalValence(),
#     "total_degree": lambda atom, i, marvin_atom: atom.GetTotalDegree(),
#     "reaction_center": lambda atom, i, marvin_atom: i == int(marvin_atom),
# }

EDGE_FEATURES = {
    "bond_type": lambda bond: bond.GetBondTypeAsDouble(),
    "is_conjugated": lambda bond: bond.GetIsConjugated(),
    "rotatable": lambda bond: chem.bond_smarts_query(bond, rotatable_bond),
}
