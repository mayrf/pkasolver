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
    """Take a PyG Dataset and return a Dataloader object. batch_size must be defined. Optional shuffle (highly discouraged) can be disabled.

    ----------
    data
        list of PyG Paired Data
    batch_size
        size of the batches set in the Dataloader function
    shuffle
        if true: shuffles the order of data in every molecule during training to prevent overfitting
    Returns
    -------
    DataLoader
        input object for training PyG Modells
    """
    return DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, follow_batch=["x_p", "x_d"]
    )


def test_ml_model(
    baseline_models: dict, X_data: np.ndarray, y_data: np.ndarray, dataset_name: str
) -> pd.DataFrame:
    """Returns a Dataframe with empirical data and data predicted from every model in the input list

    ----------
    baseline_models
        dictionary containing several models to be used to predict pKas from X_data
    X_data
        array of morgan fingerprint data - rows: molecules, columns: bits
    y_data
        empirical pKa data corresponding to molecules in X_data
    dataset_name
        name of dataset molecules in X_data and y_data belong to

    Returns
    -------
    pd.DataFrame
        Table with predicted and empirical data for every molecule in X_data and every model in baseline_models
    """

    res = {"Dataset": dataset_name, "pKa_true": y_data}
    for name, models in baseline_models.items():
        for mode, model in models.items():
            res[f"{name.upper()}_{mode}"] = model.predict(X_data[mode]).flatten()
    return pd.DataFrame(res)


def calculate_performance_of_model_on_data(
    model, loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray]:
    """

    ----------
    model
        graph model to be used for predictions
    loader
        data to be predicted

    Returns
    -------
    np.array
        list of empirical pKa values
    np.array
        list of predicted pKa values
    """

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


def predict(model, loader: DataLoader) -> np.ndarray:
    """

    ----------
    model
        graph model to be used for predictions
    loader
        data to be predicted

    Returns
    -------
    np.array
        list of predicted pKa values
    """

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


def test_graph_model(
    graph_models: dict, loader: DataLoader, dataset_name: str
) -> pd.DataFrame:
    """Returns a Dataframe with empirical data and data predicted from every graph model in the input list

    ----------
    graph_models
        dictionary containing several models to be used to predict pKas from loader
    loader
        data object containing structure and pka data for prediction
    dataset_name
        name of dataset molecules in X_data and y_data belong to

    Returns
    -------
    pd.DataFrame
        Table with predicted and empirical data for every molecule in loader and every model in graph_models
    """
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
    """

    ----------
    model
        graph model to be used for predictions
    loader
        data to be predicted

    Returns
    -------
    int
        MAE of predicted data
    int
        RMSE of predicted data
    int
        R2 of predicted data
    """

    model.to(device=DEVICE)
    x, y = calculate_performance_of_model_on_data(model, loader)
    # res["pKa_true"], res["pKa_pred"] = x, y
    MAE = mean_absolute_error(x, y)
    RMSE = np.sqrt(mean_squared_error(x, y))
    R2 = r2_score(x, y)
    # print(f"{dataset_name} - {model_name}: MAE {MAE}, RMSE {RMSE}")
    # return pd.DataFrame(res)
    return MAE, RMSE, R2
