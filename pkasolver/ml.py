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
        ref = data.reference_value
        y_dataset.extend(y_pred.tolist())
        x_dataset.extend(ref.detach().tolist())

    return np.array(x_dataset), np.array(y_dataset)


def predict_pka_value(model, loader):
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
