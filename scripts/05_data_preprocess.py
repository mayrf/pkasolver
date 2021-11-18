import argparse
import gzip
import pickle

import tqdm
from pkasolver.constants import EDGE_FEATURES, NODE_FEATURES
from pkasolver.data import (
    make_features_dicts,
    make_paired_pyg_data_from_mol,
)
from rdkit import Chem


def main(selected_node_features: dict, selected_edge_features: dict):
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
            pair_data_list = processing(
                suppl, selected_node_features, selected_edge_features
            )
    else:
        with open(args.input, "rb") as fh:
            suppl = Chem.ForwardSDMolSupplier(fh, removeHs=True)
            pair_data_list = processing(
                suppl, selected_node_features, selected_edge_features
            )

    with open(args.output, "wb") as f:
        pickle.dump(pair_data_list, f)


def processing(
    suppl, selected_node_features: dict, selected_edge_features: dict
) -> list:

    pair_data_list = []
    print("Start processing data...")

    for i, mol in tqdm.tqdm(enumerate(suppl)):
        try:
            pyg_data = make_paired_pyg_data_from_mol(
                mol, selected_node_features, selected_edge_features
            )
        except (KeyError, AssertionError) as e:
            print(e)
            continue
    print(f"PairData objects of {len(pair_data_list)} molecules successfully saved!")
    return pair_data_list


if __name__ == "__main__":
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

    main(selected_node_features, selected_edge_features)
