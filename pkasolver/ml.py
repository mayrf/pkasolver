from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from pkasolver.constants import DEVICE

# PyG Dataset to Dataloader
def dataset_to_dataloader(data, batch_size, shuffle=True):
    """Take a PyG Dataset and return a Dataloader object.
    
    batch_size must be defined.
    Optional shuffle can be enabled.
    """
    return DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, follow_batch=["x_p", "x_d"]
    )


def test_ml_model(baseline_models, X_data, y_data, dataset_name):
    res = {"Dataset": dataset_name, "pKa_true": y_data}
    for name, models in baseline_models.items():
        for mode, model in models.items():
            res[f"{name.upper()}_{mode}"] = model.predict(X_data[mode]).flatten()
    return pd.DataFrame(res)


def calculate_performance_of_model_on_data(model, loader):
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

        y_dataset.extend(y_pred.tolist())
        x_dataset.extend(data.y.tolist())

    return np.array(x_dataset), np.array(y_dataset)


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def test_graph_model(graph_models, loader, dataset_name):
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
        print(f"{dataset_name} - {model_name}: MAE {MAE}, RMSE {RMSE}")
    return pd.DataFrame(res)
