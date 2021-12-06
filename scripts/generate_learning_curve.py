import pickle

import numpy as np
import torch
from pkasolver.constants import DEVICE
from pkasolver.data import calculate_nr_of_features
from pkasolver.ml import calculate_performance_of_model_on_data, dataset_to_dataloader
from pkasolver.ml_architecture import GINPairV2, GINPairV1
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import tqdm, os

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
base_path_ = f"/data/shared/projects/pkasolver-data-clean/trained_models/training_with_{model_name}"
parameter = "hp"
prefix = "pretrained_"


def calc_mae(l: list):
    x, y = list(map(list, zip(*l)))
    return mean_absolute_error(np.array(x), np.array(y))


def calc_rmse(l: list):
    x, y = list(map(list, zip(*l)))
    return np.sqrt(mean_squared_error(np.array(x), np.array(y)))


def main():
    with open(
        "/data/shared/projects/pkasolver-data-clean/05_chembl_pretrain_data_v0.pkl",
        "rb",
    ) as f:
        train_dataset = pickle.load(f)

    if parameter == "hp":
        hidden_channels = 96
    else:
        hidden_channels = 64

    model = model_class(
        num_node_features, num_edge_features, hidden_channels=hidden_channels
    )

    nr_of_models = 500

    for j in range(9, 10):
        base_path = f"{base_path_}_v{j}_{parameter}/"
        randint = pickle.load(open(f"{base_path}/randint.pkl", "rb"))
        train_dataset_subset, validation_dataset = train_test_split(
            train_dataset, test_size=0.1, shuffle=True, random_state=randint
        )
        data = dataset_to_dataloader(train_dataset_subset, 512, shuffle=False)

        # if pickle files exist skip
        if os.path.isfile(
            f"{base_path}/{prefix}training_set_performance_{nr_of_models}.pkl"
        ):
            print(
                f"Skipping {base_path}/{prefix}training_set_performance_{nr_of_models}.pkl"
            )
        else:  # else generate data and save to base_path
            print(
                f"Calculating {base_path}/{prefix}training_set_performance_{nr_of_models}.pkl"
            )

            rmse_list = []
            mae_list = []
            for i in tqdm.tqdm(range(0, nr_of_models, 5)):  # 1_000, 5):
                checkpoint = torch.load(f"{base_path}/{prefix}model_at_{i}.pt")
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                model.to(device=DEVICE)
                x, y = calculate_performance_of_model_on_data(model, data)
                ziped_list = list(zip(x, y))
                rmse_list.append(calc_rmse(ziped_list))
                mae_list.append(calc_mae(ziped_list))
            # save to file
            res = {"rmse": rmse_list, "mae": mae_list}
            pickle.dump(
                res,
                open(
                    f"{base_path}/{prefix}training_set_performance_{nr_of_models}.pkl",
                    "wb",
                ),
            )

    for j in range(0, 10):
        base_path = f"{base_path_}_v{j}_{parameter}/"
        randint = pickle.load(open(f"{base_path}/randint.pkl", "rb"))
        train_dataset_subset, validation_dataset = train_test_split(
            train_dataset, test_size=0.1, shuffle=True, random_state=randint
        )
        data = dataset_to_dataloader(validation_dataset, 512, shuffle=False)

        # if pickle files exist skip
        if os.path.isfile(
            f"{base_path}/{prefix}validation_set_performance_{nr_of_models}.pkl"
        ):
            print(
                f"Skipping {base_path}/{prefix}validation_set_performance_{nr_of_models}.pkl"
            )
        else:  # else generate data and save to base_path
            print(
                f"Calculating {base_path}/{prefix}validation_set_performance_{nr_of_models}.pkl"
            )

            rmse_list = []
            mae_list = []
            for i in tqdm.tqdm(range(0, nr_of_models, 5)):  # 1_000, 5):
                checkpoint = torch.load(f"{base_path}/{prefix}model_at_{i}.pt")
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                model.to(device=DEVICE)
                x, y = calculate_performance_of_model_on_data(model, data)
                ziped_list = list(zip(x, y))
                rmse_list.append(calc_rmse(ziped_list))
                mae_list.append(calc_mae(ziped_list))
            # save to file
            res = {"rmse": rmse_list, "mae": mae_list}
            pickle.dump(
                res,
                open(
                    f"{base_path}/{prefix}validation_set_performance_{nr_of_models}.pkl",
                    "wb",
                ),
            )


if __name__ == "__main__":
    main()
