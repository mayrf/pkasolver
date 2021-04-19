# Import packages
from pkasolver import data
from pkasolver import chem
from pkasolver import ml
from pkasolver import stat
from pkasolver import constants as c

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Model package imports
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import GCNConv
from torch_geometric.nn import NNConv
from torch_geometric.nn import global_max_pool
from torch import optim

from captum.attr import IntegratedGradients

# Function Logic


def load_data():
    with open('data/pd_all_datasets.pkl', 'rb') as pickle_file:
        pd_dataset = pickle.load(pickle_file)

    # make pyG Dataset form 'Training'- Dataset    
    dataset = data.make_pyg_dataset(pd_dataset['Training'], node_features, edge_features, paired=paired_model)
    # Split dataset
    train_data, test_data = ml.pyg_split(dataset, train_test_split, shuffle=True)
    # Make loaders
    train_loader = ml.dataset_to_dataloader(train_data, batch_size, shuffle=True)
    test_loader = ml.dataset_to_dataloader(test_data, batch_size)




def main():
    
    
if __name__ == "__main__":
    main()