from torch_geometric.loader import DataLoader
import torch
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


def graph_predict(model, loader):
    model.eval()
    for i, data in enumerate(loader):  # Iterate in batches over the training dataset.
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

        y_true = data.y
        if i == 0:
            Y_pred = y_pred
            Y_true = y_true
        else:
            Y_true = torch.hstack((Y_true, y_true))
            Y_pred = torch.hstack((Y_pred, y_pred))
    return Y_true.numpy(), Y_pred.numpy()


def test_graph_model(graph_models, loader, dataset_name):
    res = {
        "Dataset": dataset_name,
    }
    for mode, models in graph_models.items():
        for edge, model in models.items():
            res["pKa_true"], res[f"GCN_{mode}_{edge}"] = graph_predict(model, loader)
    return pd.DataFrame(res)
