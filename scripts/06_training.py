import argparse
import os
import pickle

import torch
from pkasolver.constants import DEVICE
from pkasolver.data import calculate_nr_of_features
from pkasolver.ml import dataset_to_dataloader
from pkasolver.ml_architecture import GINProt, GINPairV3, GINPairV1, gcn_full_training

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
    parser.add_argument(
        "--reg", nargs="?", default="", help="regularization set filename"
    )
    parser.add_argument("-r", action="store_true", help="retraining run")
    parser.add_argument("--model_name", help="either GINProt, GINPairV3 or GINPairV1")
    parser.add_argument("--model", help="training directory")
    parser.add_argument(
        "--epochs",
        nargs="?",
        default="1000",
        help="set number of epochs (default=1000)",
    )
    args = parser.parse_args()
    parameter_size = "hp"

    if args.r:
        BATCH_SIZE = 64
    else:
        BATCH_SIZE = 512

    LEARNING_RATE = 0.001

    if args.model_name == "GINPairV1":
        model_name, model_class = "GINPairV1", GINPairV1
    elif args.model_name == "GINPairV3":
        model_name, model_class = "GINPairV3", GINPairV3
    elif args.model_name == "GINProt":
        model_name, model_class = "GINProt", GINProt
    else:
        raise RuntimeError()
    # where to save training progress
    print(f"Used model: {model_name}")
    if args.r:
        print("THIS IS A FINE TUNING RUN")
    # decide wheter to split training set or use explicit validation set
    print(f"load training dataset from: {args.input}")
    if args.val:
        print(f"load validation dataset from: {args.val}")
    elif args.reg:
        reg = args.reg
        print(f"load regularization dataset from: {args.reg}")
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
            with open(f"{args.model}/randint.pkl", "wb+") as f:
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
    if args.reg:
        with open(args.reg, "rb") as f:
            reg_dataset = pickle.load(f)
        reg_loader = dataset_to_dataloader(reg_dataset, 1024, shuffle=True)
    else:
        reg_loader = None

    # only load model when in retraining mode, otherwise generate new one
    if parameter_size == "hp" and args.model_name == "GINProt":
        hidden_channels = 128
    elif parameter_size == "hp" and args.model_name != "GINProt":
        hidden_channels = 96
    else:
        hidden_channels = 64
    print(f"Parameter set: {parameter_size}: {hidden_channels=}")

    model = model_class(
        num_node_features, num_edge_features, hidden_channels=hidden_channels
    )

    if args.r:
        checkpoint = torch.load(f"{args.model}/pretrained_best_model.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        if args.reg:
            prefix = "reg_everything_"
        else:
            prefix = "everything_"
        parms = []
        print("Attention: RELOADING model and extracting all layers")
        parms = model.parameters()
        optimizer = torch.optim.AdamW(parms, lr=LEARNING_RATE,)

    else:
        prefix = "pretrained_"
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,)

    model.train()
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
        reg_loader=reg_loader,
    )


if __name__ == "__main__":
    main()
