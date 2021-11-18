import argparse
import gzip
import pickle
from rdkit import Chem
from pkasolver.chem import create_conjugate
from pkasolver.data import make_features_dicts, mol_to_paired_mol_data
from pkasolver.constants import NODE_FEATURES, EDGE_FEATURES
import torch
from rdkit.Chem.AllChem import Compute2DCoords

# define selection of node and edge features
node_feat_list = [
    "element",
    "formal_charge",
    "hybridization",
    "total_num_Hs",
    "aromatic_tag",
    "total_valence",
    "total_degree",
    "is_in_ring",
    "reaction_center",
    "smarts",
]

edge_feat_list = ["bond_type", "is_conjugated", "rotatable"]
# make dicts from selection list to be used in the processing step
selected_node_features = make_features_dicts(NODE_FEATURES, node_feat_list)
selected_edge_features = make_features_dicts(EDGE_FEATURES, edge_feat_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input filename")
    parser.add_argument("--output", help="output filename")
    args = parser.parse_args()
    input_zipped = False
    print("inputfile:", args.input)
    print("outputfile:", args.output)

    # test if it's gzipped
    with gzip.open(args.input, "r") as fh:
        try:
            fh.read(1)
            input_zipped = True
        except gzip.BadGzipFile:
            input_zipped = False

    if input_zipped:
        with gzip.open(args.input, "r") as fh:
            suppl = Chem.ForwardSDMolSupplier(fh, removeHs=True)
            processing(suppl, args)
    else:
        with open(args.input, "rb") as fh:
            suppl = Chem.ForwardSDMolSupplier(fh, removeHs=True)
            processing(suppl, args)


def processing(suppl, args):
    pair_data_list = []
    print("Start processing data...")
    for i, mol in enumerate(suppl):
        if i > 5000:
            break
        # Compute2DCoords(mol)
        props = mol.GetPropsAsDict()
        try:
            pka = props["pKa"]
        except:
            print(f"No pka found for {i}. molecule: {props}")
            continue
        atom = props["marvin_atom"]
        conj = create_conjugate(mol, atom, pka)
        # sort mol and conj into protonated and deprotonated molecule
        if int(mol.GetAtomWithIdx(atom).GetFormalCharge()) > int(
            conj.GetAtomWithIdx(atom).GetFormalCharge()
        ):
            prot = mol
            deprot = conj
        else:
            prot = conj
            deprot = mol
        # create PairData object from prot and deprot with the selected node and edge features
        m = mol_to_paired_mol_data(
            prot,
            deprot,
            atom,
            selected_node_features,
            selected_edge_features,
        )
        m.y = torch.tensor(pka, dtype=torch.float32)
        m.pka_type = props["pka_number"]
        m.ID = props["ID"]
        pair_data_list.append(m)

        if i % 1000 == 0:
            if i != 0:
                print(f"{i} molecules converted")

    with open(args.output, "wb") as f:
        pickle.dump(pair_data_list, f)

    print(f"PairData objects of {len(pair_data_list)} molecules successfully saved!")


if __name__ == "__main__":
    main()
