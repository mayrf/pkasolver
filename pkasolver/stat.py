import copy

# Attribution visualisation
import random

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from scipy.linalg import block_diag
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pkasolver import ml

###############################


# def calc_importances(
#     ig, dataset, sample_size, node_feature_names, edge_feature_names=[], device="cpu"
# ):
#     """Return a DataFrame with the Attributions of all Features,
#     calculated for a number of random samples of a given dataset.
#     """
#     dataset = copy.deepcopy(dataset)
#     PAIRED = "x2" in str(ig.forward_func)
#     if "NNConv" not in str(ig.forward_func):
#         edge_feature_names = []
#     feature_names = node_feature_names + edge_feature_names
#     if PAIRED:
#         feature_names = feature_names + ["2_" + s for s in feature_names]
#     attr = np.empty((0, len(feature_names)))
#     ids = []

#     i = 0
#     for input_data in random.sample(dataset, sample_size):
#         input_data.x_p_batch = torch.zeros(input_data.x_p.shape[0], dtype=int).to(
#             device=device
#         )
#         input_data.x_d_batch = torch.zeros(input_data.x_d.shape[0], dtype=int).to(
#             device=device
#         )
#         if PAIRED and edge_feature_names == []:
#             ids.extend(
#                 [input_data.ID] * (input_data.x.shape[0] + input_data.x_d.shape[0])
#             )
#             input1 = (input_data.x_p, input_data.x_d)
#             input2 = (input_data.edge_attr_p, input_data.edge_attr_d, input_data)
#         elif PAIRED and edge_feature_names != []:
#             ids.extend(
#                 [input_data.ID]
#                 * (
#                     input_data.x_p.shape[0]
#                     + input_data.edge_index_p.shape[1]
#                     + input_data.x_d.shape[0]
#                     + input_data.edge_index_d.shape[1]
#                 )
#             )
#             input1 = (
#                 input_data.x_p,
#                 input_data.x_d,
#                 input_data.edge_attr_p,
#                 input_data.edge_attr_d,
#             )
#             input2 = input_data
#         elif not PAIRED and edge_feature_names == []:
#             ids.extend([input_data.ID] * (input_data.x_p.shape[0]))
#             input1 = input_data.x_p
#             input2 = (
#                 input_data.edge_attr_p,
#                 input_data.x_p,
#                 input_data.edge_attr_p,
#                 input_data,
#             )
#         elif not PAIRED and edge_feature_names != []:
#             ids.extend(
#                 [input_data.ID]
#                 * (input_data.x_p.shape[0] + input_data.edge_index_p.shape[1])
#             )
#             input1 = (input_data.x_p, input_data.edge_attr_p)
#             input2 = (input_data.x_d, input_data.edge_attr_d, input_data)

#         _attr = ig.attribute(
#             input1,
#             additional_forward_args=(input2),
#             internal_batch_size=input_data.x_p.shape[0],
#         )
#         if not PAIRED and edge_feature_names == []:
#             attr_row = _attr.cpu().detach().numpy()

#         else:
#             attr_row = block_diag(*[a.cpu().detach().numpy() for a in _attr])

#         attr = np.vstack((attr, attr_row))

#         if i % 10 == 0:
#             print(f"{i+1} of {sample_size}")
#         i += 1
#     df = pd.DataFrame(attr, columns=feature_names)
#     df.insert(0, "ID", ids)
#     return df


def calc_rmse(pred, true):
    return np.sqrt(mean_squared_error(pred, true))


functions = {"R^2": r2_score, "RMSE": calc_rmse, "MAE": mean_absolute_error}


def compute_stats(df, selection_col, true_col, col_exclude=[]):
    result = {}

    for dataset in df[selection_col].unique():
        x = df.loc[df[selection_col] == dataset]
        index = []
        for i, model in enumerate(
            [a for a in x.columns if a not in [selection_col, true_col] + col_exclude]
        ):
            index.append(model)
            true = x[true_col]
            pred = x[model]
            for name, func in functions.items():
                if i == 0:
                    result[(dataset, name)] = [func(true, pred).round(3)]
                else:
                    result[(dataset, name)].append(func(true, pred).round(3))
    return pd.DataFrame(result, index=index)


def cv_graph_model(graph_models, loaders):
    res = {"R^2 (mean + std)": [], "RMSE (mean + std)": [], "MAE (mean + std)": []}
    index = []
    for mode, edge_modes in graph_models.items():
        for edge, nums in edge_modes.items():
            r2 = []
            rmse = []
            mae = []
            for num, model in nums.items():
                true, pred = ml.graph_predict(model, loaders[num], device="cpu")
                r2.append(r2_score(true, pred))
                rmse.append(calc_rmse(true, pred))
                mae.append(mean_absolute_error(true, pred))
            index.append(f"GCN_{mode}_{edge}")
            res[f"R^2 (mean + std)"].append(
                f"{np.average(r2):.3f} \u00B1 {np.std(r2):.3f}"
            )
            res[f"RMSE (mean + std)"].append(
                f"{np.average(rmse):.3f} \u00B1 {np.std(rmse):.3f}"
            )
            res[f"MAE (mean + std)"].append(
                f"{np.average(mae):.3f} \u00B1 {np.std(mae):.3f}"
            )

    return pd.DataFrame(res, index=index)


def cv_ml_model(baseline_models, data):
    res = {"R^2 (mean + std)": [], "RMSE (mean + std)": [], "MAE (mean + std)": []}
    index = []
    for name, modes in baseline_models.items():
        for mode, nums in modes.items():
            r2 = []
            rmse = []
            mae = []
            for num, model in nums.items():
                true, pred = data[num]["y"], model.predict(data[num][mode]).flatten()
                r2.append(r2_score(true, pred))
                rmse.append(calc_rmse(true, pred))
                mae.append(mean_absolute_error(true, pred))
            index.append(f"{name.upper()}_{mode}")
            res[f"R^2 (mean + std)"].append(
                f"{np.average(r2):.3f} \u00B1 {np.std(r2):.3f}"
            )
            res[f"RMSE (mean + std)"].append(
                f"{np.average(rmse):.3f} \u00B1 {np.std(rmse):.3f}"
            )
            res[f"MAE (mean + std)"].append(
                f"{np.average(mae):.3f} \u00B1 {np.std(mae):.3f}"
            )
    return pd.DataFrame(res, index=index)


# taken from here:
# https://medium.com/datalab-log/measuring-the-statistical-similarity-between-two-samples-using-jensen-shannon-and-kullback-leibler-8d05af514b15
def compute_probs(data, n=10):
    h, e = np.histogram(data, n)
    p = h / data.shape[0]
    return e, p


def support_intersection(p, q):
    return list(filter(lambda x: (x[0] != 0) & (x[1] != 0), list(zip(p, q))))


def get_probs(list_of_tuples):
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q


def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))


def js_divergence(p, q):
    m = (1.0 / 2.0) * (p + q)
    return (1.0 / 2.0) * kl_divergence(p, m) + (1.0 / 2.0) * kl_divergence(q, m)


def compute_kl_divergence(train_sample, test_sample, n_bins=10):
    """Compute the KL Divergence using the support
    intersection between two different samples.
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)
    return kl_divergence(p, q)


def compute_js_divergence(train_sample, test_sample, n_bins=10):
    """
    Computes the JS Divergence using the support
    intersection between two different samples.
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)
    return js_divergence(p, q)


def plot_regression(x, y, name):
    # Define plot canvas
    g = sns.jointplot(x=x, y=y, xlim=(2, 12), ylim=(2, 12))
    # Add fit_reg lines to plot
    sns.regplot(x=x, y=y, scatter=False, ax=g.ax_joint, truncate=False)
    g.plot_marginals(sns.kdeplot, fill=True)
    # Add Diagonal line to Joint axes
    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = np.array([max(x0, y0), min(x1, y1)])
    g.ax_joint.plot(lims, lims, "-r")
    # Add error band of pka ± 1
    g.ax_joint.fill_between(lims, lims - 1, lims + 1, color="r", alpha=0.2)

    return g


def calc_stat_info(y, y_hat, name):

    stat_info = f"""
        {name}
        $r^2$ = {r2_score(y, y_hat): .2f}
        $MAE$ = {mean_absolute_error(y, y_hat): .2f}
        $RMSE$ = {calc_rmse(y, y_hat): .2f}
        $kl_div$ = {compute_kl_divergence(y_hat,y): .2f}
        """

    return stat_info


def stat_info_dict(df, x_col, y_col, datasets: list):
    info_dict = {}
    for name in datasets:
        data = df[df["Dataset"] == name]
        y, y_hat = data[x_col], data[y_col]
        info_dict[name] = calc_stat_info(y, y_hat, name)
    return info_dict
