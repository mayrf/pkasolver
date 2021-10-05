import numpy as np
import pandas as pd
from pkasolver import ml

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

###############################

# Attribution visualisation
import random
import torch
import pandas as pd
from scipy.linalg import block_diag
import copy


def calc_importances(
    ig, dataset, sample_size, node_feature_names, edge_feature_names=[], device="cpu"
):
    """Return a DataFrame with the Attributions of all Features,
    calculated for a number of random samples of a given dataset.
    """
    dataset = copy.deepcopy(dataset)
    PAIRED = "x2" in str(ig.forward_func)
    if "NNConv" not in str(ig.forward_func):
        edge_feature_names = []
    feature_names = node_feature_names + edge_feature_names
    if PAIRED:
        feature_names = feature_names + ["2_" + s for s in feature_names]
    attr = np.empty((0, len(feature_names)))
    ids = []

    i = 0
    for input_data in random.sample(dataset, sample_size):
        input_data.x_p_batch = torch.zeros(input_data.x_p.shape[0], dtype=int).to(
            device=device
        )
        input_data.x_d_batch = torch.zeros(input_data.x_d.shape[0], dtype=int).to(
            device=device
        )
        if PAIRED and edge_feature_names == []:
            ids.extend(
                [input_data.ID] * (input_data.x.shape[0] + input_data.x_d.shape[0])
            )
            input1 = (input_data.x_p, input_data.x_d)
            input2 = (input_data.edge_attr_p, input_data.edge_attr_d, input_data)
        elif PAIRED and edge_feature_names != []:
            ids.extend(
                [input_data.ID]
                * (
                    input_data.x_p.shape[0]
                    + input_data.edge_index_p.shape[1]
                    + input_data.x_d.shape[0]
                    + input_data.edge_index_d.shape[1]
                )
            )
            input1 = (
                input_data.x_p,
                input_data.x_d,
                input_data.edge_attr_p,
                input_data.edge_attr_d,
            )
            input2 = input_data
        elif not PAIRED and edge_feature_names == []:
            ids.extend([input_data.ID] * (input_data.x_p.shape[0]))
            input1 = input_data.x_p
            input2 = (
                input_data.edge_attr_p,
                input_data.x_p,
                input_data.edge_attr_p,
                input_data,
            )
        elif not PAIRED and edge_feature_names != []:
            ids.extend(
                [input_data.ID]
                * (input_data.x_p.shape[0] + input_data.edge_index_p.shape[1])
            )
            input1 = (input_data.x_p, input_data.edge_attr_p)
            input2 = (input_data.x_d, input_data.edge_attr_d, input_data)

        _attr = ig.attribute(
            input1,
            additional_forward_args=(input2),
            internal_batch_size=input_data.x_p.shape[0],
        )
        if not PAIRED and edge_feature_names == []:
            attr_row = _attr.cpu().detach().numpy()

        else:
            attr_row = block_diag(*[a.cpu().detach().numpy() for a in _attr])

        attr = np.vstack((attr, attr_row))

        if i % 10 == 0:
            print(f"{i+1} of {sample_size}")
        i += 1
    df = pd.DataFrame(attr, columns=feature_names)
    df.insert(0, "ID", ids)
    return df


from scipy import stats
import numpy as np
import pandas as pd


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
