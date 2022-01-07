import argparse
import gzip
import pickle
from itertools import repeat

import multiprocess as mp
import torch
from p_tqdm import p_umap
from pkasolver.constants import EDGE_FEATURES, NODE_FEATURES
from pkasolver.data import make_features_dicts, mol_to_paired_mol_data
from rdkit import Chem
from rdkit.Chem import PropertyMol


def main(selected_node_features: dict, selected_edge_features: dict):
    """
    takes pkl file of molecules containing pka data and returns
    pytorch geometric graph data containing
    protonated and deprotonated graphs for every pka
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input filename, type: .pkl")
    parser.add_argument("--output", help="output filename, type: .pkl")
    args = parser.parse_args()
    input_zipped = False
    print("inputfile:", args.input)
    print("outputfile:", args.output)
    print("Start processing data...")
    pair_data_list = []
    pool = mp.Pool(32)
    # test if it's gzipped
    with gzip.open(args.input, "r") as fh:
        try:
            fh.read(1)
            input_zipped = True
        except gzip.BadGzipFile:
            input_zipped = False

    if input_zipped:
        with gzip.open(args.input, "r") as fh:
            suppl = [
                PropertyMol.PropertyMol(mol)
                for mol in Chem.ForwardSDMolSupplier(fh, removeHs=True)
            ]
            print(len(suppl))
            pair_data_list.append(
                p_umap(
                    processing,
                    suppl,
                    repeat(selected_node_features),
                    repeat(selected_edge_features),
                    num_cpus=1,
                )
            )
    else:
        with open(args.input, "rb") as fh:
            suppl = pickle.load(fh)
            print(len(suppl))
            pair_data_list.extend(
                pool.starmap(
                    processing,
                    zip(
                        suppl.values(),
                        repeat(selected_node_features),
                        repeat(selected_edge_features),
                    ),
                    chunksize=100,
                )
            )

    flat_pair_data_list = [item for sublist in pair_data_list for item in sublist]
    del pair_data_list
    print(
        f"PairData objects of {len(flat_pair_data_list)} molecules successfully saved!"
    )
    with open(args.output, "wb") as f:
        pickle.dump(flat_pair_data_list, f)


def processing(
    entry, selected_node_features: dict, selected_edge_features: dict
) -> list:

    combined_mols = entry["mols"]
    pka_list = entry["pKa_list"]
    pairs = []
    for mol_pair, pka_value in zip(combined_mols, pka_list):
        chembl_id = mol_pair[0].GetProp("CHEMBL_ID")
        internal_id1 = mol_pair[0].GetProp("INTERNAL_ID")
        internal_id2 = mol_pair[1].GetProp("INTERNAL_ID")
        smiles_prop = mol_pair[0].GetProp("mol-smiles")
        smiles_deprop = mol_pair[1].GetProp("mol-smiles")
        atom_idx = mol_pair[0].GetProp("epik_atom")

        m = mol_to_paired_mol_data(
            mol_pair[0],
            mol_pair[1],
            atom_idx,
            selected_node_features,
            selected_edge_features,
        )

        m.reference_value = torch.tensor(pka_value, dtype=torch.float32)
        m.internal_id = (internal_id1, internal_id2)
        m.smiles_prop = smiles_prop
        m.smiles_deprop = smiles_deprop
        m.chembl_id = chembl_id
        m.reaction_center = atom_idx
        pairs.append(m)
    return pairs


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
