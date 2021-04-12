import numpy as np
import pandas as pd

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


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    cross_val_score,
    KFold,
)
from sklearn.metrics import make_scorer

import time


def hyperparameter_tuning(X, y, parameters: dict, test_df):
    """Take Regression variables and parameters, conduct a hyperparameter tuning and return a GridSearchCV object."""
    reg = RandomForestRegressor()

    def kl_divergence_score(y_true, y_pred):
        return compute_kl_divergence(y_pred, y_true)

    def js_divergence_score(y_true, y_pred):
        return compute_js_divergence(y_pred, y_true)

    score = {
        "r2": "r2",
        "kl-divergence": make_scorer(kl_divergence_score, greater_is_better=False),
        "js-divergence": make_scorer(js_divergence_score, greater_is_better=False),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    clf = GridSearchCV(reg, parameters, cv=kf, scoring=score, refit="r2")
    clf.fit(X, y)
    df = pd.DataFrame(clf.cv_results_)
    df.to_csv("results_hyperparameter_fitting.csv")
    return clf




###############################

# Attribution visualisation
import random
import torch
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

import matplotlib.pyplot as plt
#%matplotlib inline

def calc_importances(ig, dataset, sample_size, device='cpu'):
    """Take Integrated Gradients Object, PyG Dataset and desired sample size. 
    Return two np.arrays of importances of nodes and edges, respectivly.
    """
    attr_n = np.empty((0,dataset[0].num_features))
    attr_e = np.empty((0,dataset[0].num_edge_features))
    i = 0
    for input_data in random.sample(dataset, sample_size):
        _attr, _delta = ig.attribute((input_data.x, input_data.edge_attr),additional_forward_args=(input_data.edge_index, torch.zeros(input_data.x.shape[0], dtype=int).to(device=device)), internal_batch_size=input_data.x.shape[0], return_convergence_delta=True)
        attr_n = np.append(attr_n, _attr[0].detach().numpy(), axis=0)
        attr_e = np.append(attr_e, _attr[1].detach().numpy(), axis=0)
        if i%5==0:
            print(f'{i+1} of {sample_size}')
        i += 1
    return attr_n, attr_e

def visualize_importances(node_feature_names, edge_feature_names, node_importances, edge_importances,  title="Average Feature Importances", plot=True, axis_title="Features"):
    """Return a figure with barplots of the feature importances."""
    feature_names = node_feature_names + edge_feature_names
    importances = np.concatenate((node_importances, edge_importances))
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        fig = plt.figure(figsize=(15,6))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)
        fig.tight_layout()
        plt.show()
        plt.close()

def show_importances_distribution(node_features, node_attributions, edge_features, edge_attributions, cols=4):
    """Return a figures with the distribution of the feature importances of all tested samples."""
    i = 1
    nr_plots = len(node_features + edge_features)
    rows = nr_plots // cols + (nr_plots / cols > 0)
    plt.figure(figsize=(cols*5,rows*4))
    for n in range(len(node_features)):
        plt.subplot(rows,cols,i)
        plt.hist(node_attributions[:,n], 100);
        plt.title(f"Attribution of {node_features[n]}");
        #plt.show()
        i += 1
    for n in range(len(edge_features)):
        plt.subplot(rows,cols,i)
        plt.hist(edge_attributions[:,n], 100);
        plt.title(f"Attribution of {edge_features[n]}");
        #plt.show()
        i += 1
    plt.show()