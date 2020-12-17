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
    """
    Computes the KL Divergence using the support intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)
    return kl_divergence(p, q)


def compute_js_divergence(train_sample, test_sample, n_bins=10):
    """
    Computes the JS Divergence using the support
    intersection between two different samples
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
