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

import seaborn as sns
import numpy as np


def plot_results(x_col, y_col):
    # df = df.reset_index()
    # Define plot canvas
    g = sns.jointplot(
        x=x_col, y=y_col, xlim=(2, 12), ylim=(2, 12), kind="reg", x_bins=None
    )

    # Add fit_reg lines to plot
    # for _,gr in df.groupby(hue_col):
    sns.regplot(x=x_col, y=y_col, scatter=False, ax=g.ax_joint, truncate=False)
    # sns.kdeplot(x=x_col, y=y_col, ax=)
    # Add Diagonal line to Joint axes
    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = np.array([max(x0, y0), min(x1, y1)])
    g.ax_joint.plot(lims, lims, "-r")
    # Add error band of pka ± 1
    g.ax_joint.fill_between(lims, lims - 1, lims + 1, color="r", alpha=0.2)

    return g


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
    # test_loss = gcn_test(model, test_loader)
    MAE, RMSE, R2 = calc_testset_performace(model, test_loader)
    x, y = calculate_performance_of_model_on_data(model, test_loader)
    # print(test_loss)
    print(f"MAE: {MAE}, RMSE: {RMSE}, R2: {R2}")

    # g = plot_regression(x, y, "test")
    g = plot_results(x, y)
    g.set_axis_labels("exp", "pred")
    g.fig.suptitle(f" {model}")
    # stat_info = stat_info_dict(d, "pKa_true", model, set_list)
    # g.ax_joint.text(
    #     0.25,
    #     1,
    #     stat_info["Novartis"],
    #     size="x-small",
    #     ha="left",
    #     va="top",
    #     transform=g.ax_joint.transAxes,
    # )
    # Add stats Literature
    # g.ax_joint.text(
    #     1,
    #     0,
    #     stat_info["Literature"],
    #     size="x-small",
    #     ha="right",
    #     va="bottom",
    #     transform=g.ax_joint.transAxes,
    # )
    plt.savefig(f"test_1.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
