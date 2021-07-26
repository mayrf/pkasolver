import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

###############################

# Attribution visualisation
import random
import torch
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.linalg import block_diag
import copy

def calc_importances(ig, dataset, sample_size, node_feature_names, edge_feature_names=[], device='cpu'):
    """Return a DataFrame with the Attributions of all Features, 
    calculated for a number of random samples of a given dataset. 
    """
    dataset = copy.deepcopy(dataset)    
    PAIRED = 'x2' in str(ig.forward_func)
    if 'NNConv' not in str(ig.forward_func):
        edge_feature_names=[]
    feature_names= node_feature_names + edge_feature_names 
    if PAIRED:
        feature_names = feature_names + ['2_' + s for s in feature_names]
    attr = np.empty((0,len(feature_names)))
    ids = []
    
    i = 0
    for input_data in random.sample(dataset, sample_size):
        input_data.batch=torch.zeros(input_data.x.shape[0], dtype=int).to(device=device)
        input_data.x2_batch=torch.zeros(input_data.x2.shape[0], dtype=int).to(device=device)
        if PAIRED and edge_feature_names==[]:
            ids.extend([input_data.ID]*(input_data.x.shape[0]+input_data.x2.shape[0]))
            input1=(input_data.x,input_data.x2)
            input2=(input_data.edge_attr,input_data.edge_attr2,input_data)
        elif PAIRED and edge_feature_names!=[]:
            ids.extend([input_data.ID]*(input_data.x.shape[0]+input_data.edge_index.shape[1]+
                                        input_data.x2.shape[0]+input_data.edge_index2.shape[1]))
            input1=(input_data.x,input_data.x2,input_data.edge_attr,input_data.edge_attr2)
            input2=(input_data)
        elif not PAIRED and edge_feature_names==[]:
            ids.extend([input_data.ID]*(input_data.x.shape[0]))
            input1=(input_data.x)
            input2=(input_data.edge_attr,input_data.x2,input_data.edge_attr2,input_data)
        elif not PAIRED and edge_feature_names!=[]:
            ids.extend([input_data.ID]*(input_data.x.shape[0]+input_data.edge_index.shape[1]))
            input1=(input_data.x, input_data.edge_attr)
            input2=(input_data.x2,input_data.edge_attr2,input_data)
            
        
        _attr = ig.attribute(input1,
                             additional_forward_args=(input2), 
                             internal_batch_size=input_data.x.shape[0])
        if not PAIRED and edge_feature_names==[]:
            attr_row = _attr.detach().numpy()
            
        else:
            attr_row = block_diag(*[a.detach().numpy() for a in _attr])

        attr = np.vstack((attr, attr_row))
        
        if i%10==0:
            print(f'{i+1} of {sample_size}')
        i += 1
    df =pd.DataFrame(attr,columns=feature_names)
    df.insert(0, 'ID', ids)
    return df


from scipy import stats
import numpy as np
import pandas as pd

def compute_stats(df, selection_col, true_col,col_exclude=[]):
    result ={}
    
    for dataset in df[selection_col].unique():
        x = df.loc[df[selection_col]== dataset]
        index = []
        for i, model in enumerate([a for a in x.columns if a not in [selection_col, true_col]+col_exclude]):
            index.append(model)
            true = x[true_col]
            pred = x[model]
            for name,func in functions.items():
                if i == 0:
                    result[(dataset,name)]=[func(true,pred).round(3)]
                else:
                    result[(dataset,name)].append(func(true,pred).round(3))
    return pd.DataFrame(result, index=index)





# def test_model(model, loader, dataset_name, DEVICE='cpu'):
#     """Calculate the prediction values of all the sample in a given DataLoader object
#     and return them together with the real values in a Dataframe
#     """
#     model.eval()
#     i = 0
#     for data in loader:  # Iterate in batches over the training dataset.
#         data.to(device=DEVICE)
#         y_pred = model(x=data.x, x2=data.x2,edge_attr=data.edge_attr, edge_attr2=data.edge_attr2, data=data).reshape(-1)
#         y_true = data.y
#         if i == 0:
#             Y_pred = y_pred 
#             Y_true = y_true
#         else:
#             Y_true=torch.hstack((Y_true,y_true))
#             Y_pred=torch.hstack((Y_pred,y_pred))
#         i+=1
#     return pd.DataFrame({'Dataset':dataset_name, 'pKa_true':Y_true.detach().numpy(), 'pKa_exp':Y_pred.detach().numpy()})

# def rmse(pred, true):
#     return np.sqrt(mean_squared_error(pred, true))
    
# functions = {
#     'R^2':r2_score,
#     'RMSE':rmse,
#     'MAE':mean_absolute_error
# }
    


# def bootstrap(func, pred, true):
#     true = np.array(true)
#     pred = np.array(pred)
#     length = len(pred)
#     samples = []
#     results = []
#     for i in range(10000):
#         samples.append(np.random.choice(list(range(length)), replace=True, size=length//2)) 
#     for sample in samples:
#         results.append(func(pred[sample], true[sample]))
#     return np.percentile(results, 5), np.percentile(results, 50), np.percentile(results, 95)


# # taken from here:
# # https://medium.com/datalab-log/measuring-the-statistical-similarity-between-two-samples-using-jensen-shannon-and-kullback-leibler-8d05af514b15
# def compute_probs(data, n=10):
#     h, e = np.histogram(data, n)
#     p = h / data.shape[0]
#     return e, p


# def support_intersection(p, q):
#     return list(filter(lambda x: (x[0] != 0) & (x[1] != 0), list(zip(p, q))))


# def get_probs(list_of_tuples):
#     p = np.array([p[0] for p in list_of_tuples])
#     q = np.array([p[1] for p in list_of_tuples])
#     return p, q


# def kl_divergence(p, q):
#     return np.sum(p * np.log(p / q))


# def js_divergence(p, q):
#     m = (1.0 / 2.0) * (p + q)
#     return (1.0 / 2.0) * kl_divergence(p, m) + (1.0 / 2.0) * kl_divergence(q, m)


# def compute_kl_divergence(train_sample, test_sample, n_bins=10):
#     """Compute the KL Divergence using the support
#     intersection between two different samples.
#     """
#     e, p = compute_probs(train_sample, n=n_bins)
#     _, q = compute_probs(test_sample, n=e)

#     list_of_tuples = support_intersection(p, q)
#     p, q = get_probs(list_of_tuples)
#     return kl_divergence(p, q)


# def compute_js_divergence(train_sample, test_sample, n_bins=10):
#     """
#     Computes the JS Divergence using the support
#     intersection between two different samples.
#     """
#     e, p = compute_probs(train_sample, n=n_bins)
#     _, q = compute_probs(test_sample, n=e)

#     list_of_tuples = support_intersection(p, q)
#     p, q = get_probs(list_of_tuples)
#     return js_divergence(p, q)


# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import (
#     GridSearchCV,
#     train_test_split,
#     cross_val_score,
#     KFold,
# )
# from sklearn.metrics import make_scorer

# import time


# def hyperparameter_tuning(X, y, parameters: dict, test_df):
#     """Take Regression variables and parameters, conduct a hyperparameter tuning and return a GridSearchCV object."""
#     reg = RandomForestRegressor()

#     def kl_divergence_score(y_true, y_pred):
#         return compute_kl_divergence(y_pred, y_true)

#     def js_divergence_score(y_true, y_pred):
#         return compute_js_divergence(y_pred, y_true)

#     score = {
#         "r2": "r2",
#         "kl-divergence": make_scorer(kl_divergence_score, greater_is_better=False),
#         "js-divergence": make_scorer(js_divergence_score, greater_is_better=False),
#     }

#     kf = KFold(n_splits=5, shuffle=True, random_state=0)
#     clf = GridSearchCV(reg, parameters, cv=kf, scoring=score, refit="r2")
#     clf.fit(X, y)
#     df = pd.DataFrame(clf.cv_results_)
#     df.to_csv("results_hyperparameter_fitting.csv")
#     return clf