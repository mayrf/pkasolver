import argparse
import gzip
import pickle

from p_tqdm import p_umap
from pkasolver.constants import EDGE_FEATURES, NODE_FEATURES
from pkasolver.data import (
    make_features_dicts,
    make_paired_pyg_data_from_mol,
)
from rdkit import Chem
import multiprocessing as mp
from itertools import repeat


def main(selected_node_features: dict, selected_edge_features: dict):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input filename")
    parser.add_argument("--output", help="output filename")
    args = parser.parse_args()
    input_zipped = False
    print("inputfile:", args.input)
    print("outputfile:", args.output)
    print("Start processing data...")
    pair_data_list = []

    # test if it's gzipped
    with gzip.open(args.input, "r") as fh:
        try:
            fh.read(1)
            input_zipped = True
        except gzip.BadGzipFile:
            input_zipped = False

    if input_zipped:
        with gzip.open(args.input, "r") as fh:
            suppl = [mol for mol in Chem.ForwardSDMolSupplier(fh, removeHs=True)]
            pair_data_list.append(
                p_umap(
                    processing,
                    suppl,
                    repeat(selected_node_features),
                    repeat(selected_edge_features),
                    num_cpus=2,
                )
            )
    else:
        with open(args.input, "rb") as fh:
            suppl = [mol for mol in Chem.ForwardSDMolSupplier(fh, removeHs=True)]
            pair_data_list = processing(
                suppl, selected_node_features, selected_edge_features
            )

    print(f"PairData objects of {len(pair_data_list)} molecules successfully saved!")
    with open(args.output, "wb") as f:
        pickle.dump(pair_data_list, f)


def processing(mol, selected_node_features: dict, selected_edge_features: dict) -> list:

    pair_data_list = []
    if not mol:
        return None
    
    try:
        pyg_data = make_paired_pyg_data_from_mol(
            mol, selected_node_features, selected_edge_features
        )
        return pyg_data

    except (KeyError, AssertionError) as e:
        print(e)
        return None


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
