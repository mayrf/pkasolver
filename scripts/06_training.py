import argparse
import os
import pickle

import torch
from pkasolver.constants import DEVICE
from pkasolver.data import calculate_nr_of_features
from pkasolver.ml import dataset_to_dataloader
from pkasolver.ml_architecture import GINPairV1, gcn_full_training

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="training set filename")
    parser.add_argument("--val", nargs="?", default="", help="validation set filename")
    parser.add_argument("-r", action="store_true", help="retraining run")
    parser.add_argument("--model", help="training directory")
    parser.add_argument(
        "--epochs",
        nargs="?",
        default="1000",
        help="set number of epochs (default=1000)",
    )
    args = parser.parse_args()

    if args.r:
        BATCH_SIZE = 64
    else:
        BATCH_SIZE = 512

    LEARNING_RATE = 0.001

    model_name, model_class = "GINPairV1", GINPairV1

    # where to save training progress

    # decide wheter to split training set or use explicit validation set
    print(f"load training dataset from: {args.input}")
    if args.val:
        print(f"load validation dataset from: {args.val}")
    else:
        print(f"random 90:10 split is used to generate validation set.")

    print(f"Write models and training progress to: {args.model}")
    os.makedirs(args.model, exist_ok=True)

    # read training set
    with open(args.input, "rb") as f:
        train_dataset = pickle.load(f)

    NUM_EPOCHS = int(args.epochs)
    print(f"number of epochs set to {NUM_EPOCHS}")

    # if validation argument is not specified randomly split training set
    if not args.val:
        from sklearn.model_selection import train_test_split
        import random

        if os.path.isfile(f"{args.model}/randint.pkl"):
            rs = pickle.load(open(f"{args.model}/randint.pkl", "rb"))
            print(f"Loading randing: {rs}")
        else:
            rs = random.randint(
                0, 1_000_000
            )  # save random_state to reproduce splitting if needed!
            print(rs)
            with open(f"{args.model}/randint_.pkl", "wb+") as f:
                pickle.dump(rs, f)

        train_dataset, validation_dataset = train_test_split(
            train_dataset, test_size=0.1, shuffle=True, random_state=rs
        )
    else:
        # if validation set is specified load it
        with open(args.val, "rb") as f:
            validation_dataset = pickle.load(f)

    train_loader = dataset_to_dataloader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = dataset_to_dataloader(validation_dataset, BATCH_SIZE, shuffle=True)

    # only load model when in retraining mode, otherwise generate new one
    model = model_class(num_node_features, num_edge_features, hidden_channels=64)

    if args.r:
        checkpoint = torch.load(f"{args.model}/pretrained_best_model.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        prefix = "retrained_"
        print("Attention: RELOADING model and freezing GNN")
        print("Freeze Convs submodule parameter.")
        print(model.get_submodule("convs"))
        for p in model.get_submodule("convs"):
            p.requires_grad = False
        print("FROZEN!")
    else:
        prefix = "pretrained_"

    model.train()
    # only use models that are not frozen in optimization
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE,
    )
    print(
        "Number of parameters: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad == True),
    )

    print(f'Training {model_name} at epoch {model.checkpoint["epoch"]} ...')
    print(f"LR: {LEARNING_RATE}")
    print(f"Batch-size: {BATCH_SIZE}")
    print(model_name)
    print(model)
    print(f"Training on {DEVICE}.")
    print(f"Saving models to: {args.model}")
    results = gcn_full_training(
        model.to(device=DEVICE),
        train_loader,
        val_loader,
        optimizer,
        NUM_EPOCHS=NUM_EPOCHS,
        path=args.model,
        prefix=prefix,
    )


if __name__ == "__main__":
    main()
