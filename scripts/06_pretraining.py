import argparse
import pickle
import os
import torch
from torch import optim
import tqdm
from pkasolver.constants import EDGE_FEATURES, NODE_FEATURES
from pkasolver.ml import dataset_to_dataloader
from pkasolver.data import calculate_nr_of_features
from pkasolver.ml_architecture import GINProt, gcn_full_training

from pkasolver.constants import DEVICE, SEED

BATCH_SIZE = 64
NUM_EPOCHS = 1400
LEARNING_RATE = 0.001

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
num_node_features = calculate_nr_of_features(node_feat_list)
num_edge_features = calculate_nr_of_features(edge_feat_list)

model_name, model_class = "GINProt", GINProt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input filename")
    parser.add_argument("--val", help="validation filename")
    parser.add_argument("--output", help="output filename")
    args = parser.parse_args()
    input_zipped = False
    print("inputfile:", args.input)
    print("outputfile:", args.output)
    with open(args.input, "rb") as f:
        dataset = pickle.load(f)
    train_loader = dataset_to_dataloader(dataset, BATCH_SIZE, shuffle=True)
    with open(args.val, "rb") as f:
        dataset = pickle.load(f)
    val_loader = dataset_to_dataloader(dataset, BATCH_SIZE, shuffle=True)

    if os.path.isfile(args.output):
        print("Attention: RELOADING model")
        with open(args.output, "rb") as pickle_file:
            model = pickle.load(pickle_file)
    else:
        model = model_class(num_node_features, num_edge_features, hidden_channels=96)
    if model.checkpoint["epoch"] < NUM_EPOCHS:
        model.to(device=DEVICE)
        print(model.checkpoint["epoch"])
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
        try:
            optimizer.load_state_dict(model.checkpoint["optimizer_state"])
        except:
            pass
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, verbose=True
        )

        print(f'Training {model_name} at epoch {model.checkpoint["epoch"]} ...')
        print(f"LR: {LEARNING_RATE}")
        print(model_name)
        print(model)
        print(f"Training on {DEVICE}.")

        results = gcn_full_training(
            model.to(device=DEVICE),
            train_loader,
            val_loader,
            optimizer,
            args.output,
            NUM_EPOCHS,
        )

        with open(args.output, "wb") as pickle_file:
            pickle.dump(model.to(device="cpu"), pickle_file)
        print(f"trained/loaded gcn models successfully")


if __name__ == "__main__":
    main()
