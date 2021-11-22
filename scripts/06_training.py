import argparse
import pickle

import torch
from pkasolver.constants import DEVICE
from pkasolver.data import calculate_nr_of_features
from pkasolver.ml import dataset_to_dataloader
from pkasolver.ml_architecture import GINPairV2, gcn_full_training


def main():
    BATCH_SIZE = 512
    NUM_EPOCHS = 2000
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

    model_name, model_class = "GINPairV2", GINPairV2

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="training set filename")
    parser.add_argument("--val", nargs="?", default="", help="validation set filename")
    parser.add_argument("-r", action="store_true")
    parser.add_argument("--model", help="trained model filename")
    parser.add_argument("--epochs", help="set number of epochs")
    args = parser.parse_args()

    # decide wheter to split training set or use explicit validation set
    print(f"load training dataset from: {args.input}")
    if args.val:
        print(f"load validation dataset from: {args.val}")
    else:
        print(f"random 90:10 split is used to generate validation set.")
    print(f"Write finished model to: {args.model}")

    # read training set
    with open(args.input, "rb") as f:
        train_dataset = pickle.load(f)

    if args.epochs:
        NUM_EPOCHS = int(args.epochs)
        print(f"number of epochs set to {NUM_EPOCHS}")
    else:
        print(f"number of epochs set to {NUM_EPOCHS} (default)")

    # if validation argument is not specified randomly split training set
    if not args.val:
        from sklearn.model_selection import train_test_split

        train_dataset, validation_dataset = train_test_split(
            train_dataset, test_size=0.1, shuffle=True
        )
    else:
        # if validation set is specified load it
        with open(args.val, "rb") as f:
            validation_dataset = pickle.load(f)

    train_loader = dataset_to_dataloader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = dataset_to_dataloader(validation_dataset, BATCH_SIZE, shuffle=True)

    # only load model when in retraining mode, otherwise generate new one
    if args.r:
        print("Attention: RELOADING model and freezing GNN")
        with open(args.model, "rb") as pickle_file:
            model = pickle.load(pickle_file)
    else:
        model = model_class(num_node_features, num_edge_features, hidden_channels=96)

    if model.checkpoint["epoch"] < NUM_EPOCHS:
        model.to(device=DEVICE)
        print(model.checkpoint["epoch"])
        print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
        if args.r:
            print("Freeze Convs submodule parameter.")
            print(model.get_submodule("convs"))
            for p in model.get_submodule("convs"):
                p.requires_grad = False
            print("FROZEN!")

        # only use models that are not frozen in optimization
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE,
        )
        print(
            "Number of parameters: ",
            sum(p.numel() for p in model.parameters() if p.requires_grad == True),
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
            NUM_EPOCHS=NUM_EPOCHS,
        )

        if args.r:
            fully_trained_model = args.model.split(".")[0] + "_fully_trained_fritz.pkl"
        else:
            fully_trained_model = args.model

        with open(fully_trained_model, "wb") as pickle_file:
            pickle.dump(model.to(device="cpu"), pickle_file)
        print(f"trained gcn models is saved to: {fully_trained_model}")


if __name__ == "__main__":
    main()
