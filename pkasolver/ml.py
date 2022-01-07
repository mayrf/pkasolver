from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader

from pkasolver.constants import DEVICE


# PyG Dataset to Dataloader
def dataset_to_dataloader(
    data: list, batch_size: int, shuffle: bool = True
) -> DataLoader:
    """Take a PyG Dataset and return a Dataloader object.

    batch_size must be defined.
    Optional shuffle can be enabled.
    """
    return DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, follow_batch=["x_p", "x_d"]
    )


def test_ml_model(
    baseline_models: dict, X_data: np.ndarray, y_data: np.ndarray, dataset_name: str
) -> pd.DataFrame:
    res = {"Dataset": dataset_name, "pKa_true": y_data}
    for name, models in baseline_models.items():
        for mode, model in models.items():
            res[f"{name.upper()}_{mode}"] = model.predict(X_data[mode]).flatten()
    return pd.DataFrame(res)


def calculate_performance_of_model_on_data(
    model, loader
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_dataset, x_dataset = [], []
    for data in loader:  # Iterate in batches over the training dataset.

        data.to(device=DEVICE)
        y_pred = (
            model(
                x_p=data.x_p,
                x_d=data.x_d,
                edge_attr_p=data.edge_attr_p,
                edge_attr_d=data.edge_attr_d,
                data=data,
            )
            .reshape(-1)
            .detach()
        )
        ref = data.reference_value
        y_dataset.extend(y_pred.tolist())
        x_dataset.extend(ref.detach().tolist())

    return np.array(x_dataset), np.array(y_dataset)


def predict(model, loader) -> np.ndarray:
    model.eval()
    results = []
    for data in loader:  # Iterate in batches over the training dataset.
        data.to(device=DEVICE)
        y_pred = (
            model(
                x_p=data.x_p,
                x_d=data.x_d,
                edge_attr_p=data.edge_attr_p,
                edge_attr_d=data.edge_attr_d,
                data=data,
            )
            .reshape(-1)
            .detach()
        )

        results.extend(y_pred.tolist())

    return np.array(results)


# def calculate_performance_of_model_on_data_old(
#     model, loader
# ) -> Tuple[np.ndarray, np.ndarray]:
#     model.eval()
#     y_dataset, x_dataset = [], []
#     for data in loader:  # Iterate in batches over the training dataset.

#         data.to(device=DEVICE)
#         y_pred = (
#             model(
#                 x_p=data.x_p,
#                 x_d=data.x_d,
#                 edge_attr_p=data.edge_attr_p,
#                 edge_attr_d=data.edge_attr_d,
#                 data=data,
#             )
#             .reshape(-1)
#             .detach()
#         )

#         y_dataset.extend(y_pred.tolist())
#         x_dataset.extend(data.y.tolist())

#     return np.array(x_dataset), np.array(y_dataset)


def test_graph_model(graph_models, loader, dataset_name: str) -> pd.DataFrame:
    res = {
        "Dataset": dataset_name,
    }

    for model_name in graph_models:
        model = graph_models[model_name]
        model.to(device=DEVICE)
        x, y = calculate_performance_of_model_on_data(model, loader)
        res["pKa_true"], res[f"{model_name}"] = x, y
        MAE = mean_absolute_error(x, y)
        RMSE = np.sqrt(mean_squared_error(x, y))
        # print(f"{dataset_name} - {model_name}: MAE {MAE}, RMSE {RMSE}")
    return pd.DataFrame(res)


def calc_testset_performace(model, loader) -> Tuple[int, int, int]:

    model.to(device=DEVICE)
    x, y = calculate_performance_of_model_on_data(model, loader)
    # res["pKa_true"], res["pKa_pred"] = x, y
    MAE = mean_absolute_error(x, y)
    RMSE = np.sqrt(mean_squared_error(x, y))
    R2 = r2_score(x, y)
    # print(f"{dataset_name} - {model_name}: MAE {MAE}, RMSE {RMSE}")
    # return pd.DataFrame(res)
    return MAE, RMSE, R2
