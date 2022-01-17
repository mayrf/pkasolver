from typing import Tuple

import numpy as np
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


