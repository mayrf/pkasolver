import argparse
import pickle
import torch

from pkasolver.constants import DEVICE
from pkasolver.data import calculate_nr_of_features
from pkasolver.ml import dataset_to_dataloader, calc_testset_performace
from pkasolver.ml_architecture import GINPairV1, gcn_test

BATCH_SIZE = 512
NUM_EPOCHS = 20
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

model_name, model_class = "GINPairV1", GINPairV1


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="location where models are stored")
    parser.add_argument("--testset", help="test set filename")
    args = parser.parse_args()

    # decide wheter to split training set or use explicit validation set
    print(f"load test dataset from: {args.testset}")

    # read training set
    with open(args.testset, "rb") as f:
        test_dateset = pickle.load(f)

    test_loader = dataset_to_dataloader(test_dateset, BATCH_SIZE, shuffle=False)
    model = model_class(num_node_features, num_edge_features, hidden_channels=96)
    checkpoint = torch.load(f"{args.model}/retrained_best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    model.to(device=DEVICE)
    test_loss = gcn_test(model, test_loader)
    MAE, RMSE, R2 = calc_testset_performace(model, test_loader)
    # print(test_loss)
    print(f"MAE: {MAE}, RMSE: {RMSE}, R2: {R2}")


if __name__ == "__main__":
    main()
