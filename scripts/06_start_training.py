import argparse
import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from pkasolver.data import (
    calculate_nr_of_features,
    load_data,
    make_pyg_dataset_from_dataframe,
    preprocess_all,
    train_validation_set_split,
)
from pkasolver.ml import dataset_to_dataloader
from pkasolver.ml_architecture import (
    GINPairV2,
    gcn_full_training,
)
from pkasolver.constants import DEVICE

plt.rcParams["figure.figsize"] = (6.25, 6.25)
sns.set_theme(style="ticks")

import torch
from torch import optim

LEARNING_RATE = 0.001
NUM_EPOCHS = 3000
PRETRAINING = True
BATCH_SIZE = 512
NUM_GRAPH_LAYERS = 4
NUM_LINEAR_LAYERS = 2
HIDDEN_CHANNELS = 96

list_node_features = [
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
list_edge_features = ["bond_type", "is_conjugated", "rotatable"]


def plot_results(results, title):

    plt.plot(results["training-set"], label="training set")
    plt.plot(results["training-set"], label="training set")
    plt.plot(results["validation-set"], label="validation set")
    plt.title(title)
    plt.ylim([0, 4])
    plt.ylim([0, 4])
    plt.show()


def data_preprocessing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input filename")
    parser.add_argument("--output", help="output filename")

    args = parser.parse_args()

    print("inputfile:", args.input)
    print("outputfile:", args.output)

    print("Generating data ...")
    sdf_paths = load_data("/home/mwieder/Work/Projects/pkasolver/data/Baltruschat/")
    dataset_dfs = preprocess_all(sdf_paths)

    if not PRETRAINING:
        (
            dataset_dfs["experimental_training_set"],
            dataset_dfs["experimental_validation_set"],
        ) = train_validation_set_split(
            dataset_dfs["Training"], 0.8, 42
        )  # take a copy of the "Training" dataset, shuffle and split it into train and validation datasets and store them as Dataframe in respective dict

    del dataset_dfs["Training"]
    print("Generating data ...")
    graph_data = {}
    for name, df in dataset_dfs.items():
        print(f"Generating data for: {name}")
        graph_data[name] = make_pyg_dataset_from_dataframe(
            df, list_node_features, list_edge_features, paired=True
        )

    # create an iterable loader object from the list of graph data of each dataset and store them in a dictonary
    loaders = {}
    for name, dataset in graph_data.items():
        print(f"Generating loader for {name}")
        if name == "train_split":
            loaders[name] = dataset_to_dataloader(
                dataset, BATCH_SIZE, shuffle=True
            )  # Shuffling is essential to avoid overfitting on particular batches
        else:
            loaders[name] = dataset_to_dataloader(
                dataset, BATCH_SIZE, shuffle=False
            )  # Testsets must not be shuffled in order to be able to calculate per datapoint predcitons with all graph and baselinemodels in the analysis part

    if PRETRAINING:
        dataset = pickle.load(open(args.input, "rb"))
        from sklearn.model_selection import train_test_split

        train_set, validation_set = train_test_split(
            dataset, test_size=0.1, shuffle=True
        )
        loaders["epik_training_set"] = dataset_to_dataloader(
            train_set, BATCH_SIZE, shuffle=True
        )
        loaders["epik_validation_set"] = dataset_to_dataloader(
            validation_set, BATCH_SIZE, shuffle=True
        )
    return loaders


def training(loaders):

    if PRETRAINING:
        training_set = "epik_training_set"
        validation_set = "epik_validation_set"
    else:
        training_set = "experimental_training_set"
        validation_set = "experimental_validation_set"

    models = [
        ("GINPairV2", GINPairV2),
    ]

    num_node_features = calculate_nr_of_features(list_node_features)
    num_edge_features = calculate_nr_of_features(list_edge_features)
    print(f"Nr of node features: {num_node_features}")
    print(f"Nr of edge features: {num_edge_features}")

    for model_name, model_class in models:

        path = f"pretrained_models/"
        os.makedirs(path, exist_ok=True)
        pkl_file_name = f"{path}/{model_name}.pkl"

        if os.path.isfile(pkl_file_name):
            print("Attention: RELOADING model")
            with open(pkl_file_name, "rb") as pickle_file:
                model = pickle.load(pickle_file)
        else:
            model = model_class(
                num_node_features, num_edge_features, hidden_channels=64
            )

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
            print(f"Training set used: {training_set}")
            print(f"Validation set used: {validation_set}")
            results = gcn_full_training(
                model.to(device=DEVICE),
                loaders[training_set],
                loaders[validation_set],
                optimizer,
            )

            plot_results(results, f"{model_name}")
            with open(f"{path}/{model_name}.pkl", "wb") as pickle_file:
                pickle.dump(model.to(device="cpu"), pickle_file)


def main():
    print("#####################################")
    if PRETRAINING:
        print("THIS IS A PRETRAINING RUN")
    else:
        print("THIS IS A TRANSFER LEARNING RUN")
    print("#####################################")
    loaders = data_preprocessing()
    training(loaders)


if __name__ == "__main__":
    main()
