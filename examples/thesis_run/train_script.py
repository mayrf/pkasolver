from config import *
from architecture import *

from pkasolver import data, chem, ml, stat
from pkasolver import constants as c

import pickle
import os
import sys
import random 
random.seed(SEED)
import numpy as np
import pandas as pd
import seaborn as sns
import copy
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from torch import optim
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("runs/pka")

#script variables
if sys.argv[4] == 'None':
    run_cv=sys.argv[4] # 0,1,2,3,4
else:
    run_cv=int(sys.argv[4]) # 0,1,2,3,4

run_edge= str(sys.argv[3]).lower() # ['no-edge', 'edge']
run_mode=str(sys.argv[2]).lower()      # ['prot','deprot','pair']
cv_model=str(sys.argv[1]).upper()        # ['gcn','rfr','pls','mlr']

#makes sure the directory for saving the run data exists
os.makedirs('run_data/', exist_ok=True)

#checks if the finished dictonary of Dataframes already exist
if os.path.isfile('run_data/data_dfs.pkl'):
    with open('run_data/data_dfs.pkl', 'rb') as pickle_file:
        data_dfs = pickle.load(pickle_file)
        
#creates DataFrames for all datasets, stores them in a dictonary and saves it as as .pkl in in the run_data folder  
else:
    data_paths = {'Training':SDF_TRAIN}
    for path, name in zip(SDF_TEST,TEST_NAMES):
        data_paths[name]=path

    data_dfs = data.preprocess_all(data_paths, title='pd_all_datasets')
    data_dfs['train_split'], data_dfs['val_split'] = data.train_test_split_df(data_dfs['Training'], TRAIN_TEST_SPLIT)
    data_dfs.pop('Training')

    os.makedirs('run_data/', exist_ok=True)
    with open('run_data/data_dfs.pkl', 'wb') as pickle_file:
        pickle.dump(data_dfs,pickle_file)
            
print(data_dfs.keys())

if os.path.isfile('run_data/fp_data.pkl'):
    with open('run_data/fp_data.pkl', 'rb') as pickle_file:
        fp_data = pickle.load(pickle_file)
        
else:
    fp_data = {}
    i = 1
    for name, df in data_dfs.items():
        X_feat, y = data.make_stat_variables(df, [],["pKa"])
        X_prot = chem.morgan_fp_array(df, 'protonated', nBits=FP_BITS, radius=FP_RADIUS, useFeatures=True )
        X_deprot= chem.morgan_fp_array(df, 'deprotonated', nBits=FP_BITS, radius=FP_RADIUS, useFeatures=True)
        X = np.concatenate((X_prot, X_deprot), axis=1)
        fp_data[f'{name}']={
            'prot':X_prot,
            'deprot':X_deprot,
            'pair':X,
            'y':y
        }
        i += 1

    with open('run_data/fp_data.pkl', 'wb') as pickle_file:
        pickle.dump(fp_data,pickle_file)

    #add max tanimotosimilarity to data_dfs of external test sets
    train_name= 'train_split'
    val_name='val_split'
    for name, dataset in fp_data.items():
        if name in [train_name, val_name]:
            pass
        else:
            print(f'calculating similarities for {name}')
            max_scores=[]
            for test_mol in dataset['prot']:
                scores=[]
                for ref_mol in fp_data[train_name]['prot']:
                    scores.append(chem.tanimoto(test_mol, ref_mol))
                max_scores.append(max(scores))
            data_dfs[name]['Similarity_max'] = max_scores
    
    with open('run_data/data_dfs.pkl', 'wb') as pickle_file:
        pickle.dump(data_dfs,pickle_file)
    
            
print('fp_data keys:',fp_data.keys())

if os.path.isfile('run_data/graph_data.pkl'):
    with open('run_data/graph_data.pkl', 'rb') as pickle_file:
        graph_data = pickle.load(pickle_file)

else:
    graph_data = {}
    for name, df in data_dfs.items():
        graph_data[name]= data.make_pyg_dataset(df, NODE_FEATURES, EDGE_FEATURES, pair=True)
        
    with open('run_data/graph_data.pkl', 'wb') as pickle_file:
        pickle.dump(graph_data,pickle_file)

print('graph_data keys:',graph_data.keys())        

loaders = {}
for name, dataset in graph_data.items():
    loaders[name] = ml.dataset_to_dataloader(dataset, BATCH_SIZE)

print('loaders keys:',loaders.keys())  


mol_modes=[run_mode]
edge_modes=[run_edge]
cv = [run_cv]
            
    
if cv_model == 'GCN' and run_cv != 'None':

    cv_graph_data = data.slice_list(graph_data['train_split']+graph_data['val_split'],5)
    cv_graph_train, cv_graph_val = data.cross_val_lists(cv_graph_data,run_cv)

    cv_loaders={'train':{},'val':{}}
    for i in range(5):
        train, val = data.cross_val_lists(cv_graph_data,run_cv)
        cv_loaders['train'][i] = ml.dataset_to_dataloader(train, BATCH_SIZE)
        cv_loaders['val'][i] = ml.dataset_to_dataloader(val, BATCH_SIZE)

    
    graph_models_cv = {}
    for mode in mol_modes:
        graph_models_cv[mode] ={}
        for edge in edge_modes:
            graph_models_cv[mode][edge] = {}
            for num_cv in cv: 
                path = f'cv_models/gcn/{mode}/{edge}/{num_cv}/'
                if os.path.isfile(path+'model.pkl'):
                    with open(path+'model.pkl', 'rb') as pickle_file:
                        graph_models_cv[mode][edge][num_cv] = pickle.load(pickle_file)
                    model = graph_models_cv[mode][edge][num_cv]
                else:
                    ef = edge == 'edge'
                    if mode == 'pair':
                        model = GCN_paired(edge_conv=ef).to(device=DEVICE)
                    elif mode == 'deprot':
                        model = GCN_deprot(edge_conv=ef).to(device=DEVICE)
                    else:
                        model = GCN_prot(edge_conv=ef).to(device=DEVICE)
                    graph_models_cv[mode][edge][num_cv] = model
                    os.makedirs(path, exist_ok=True)
                if model.checkpoint['epoch'] < NUM_EPOCHS:   
                    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                    try:
                        optimizer.load_state_dict(model.checkpoint['optimizer_state'])    
                    except:
                        pass
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
                    print(f'Training GCN_{mode} with {edge}...')
                    gcn_full_training(model,cv_loaders['train'][num_cv],cv_loaders['val'][num_cv], optimizer, path)

                    with open(path+'model.pkl', 'wb') as pickle_file:
                        pickle.dump(model,pickle_file)
            


#create dictionary with modes as keys and a list of 5 arrays for each value
cv_fp_data={}
for name, array in fp_data['train_split'].items():
    try:
        cv_fp_data[name]=np.vstack((array,fp_data['val_split'][name]))
    except:
        cv_fp_data[name]=np.hstack((array,fp_data['val_split'][name]))
for name, array in cv_fp_data.items():
    cv_fp_data[name] = data.slice_list(array,5)

#generate 
cv_fp_sets={'train':{},'val':{}}
for i in range(5):
    cv_fp_train={}
    cv_fp_val={}
    for name, array in cv_fp_data.items():
        train, val = data.cross_val_lists(array,i)
        cv_fp_train[name] = np.array(train, dtype=object)
        cv_fp_val[name] = np.array(val, dtype=object)
    cv_fp_sets['train'][i]=cv_fp_train
    cv_fp_sets['val'][i]=cv_fp_val
    
cv_models = [cv_model]
cv_mode = [run_mode]


models_dict = {
#         'MLR':LinearRegression(),
        'RFR':RandomForestRegressor(n_estimators=NUM_ESTIMATORS, random_state=SEED),  #Baltruschat n_estimatores = 1000
        'PLS':PLSRegression()
    }

baseline_models_cv = {}
train_name= 'train_split'
val_name='val_split'

# baseline cv training

if cv_model in ['RFR','PLS','MLR'] and run_cv != 'None':
    for name in cv_models:
        baseline_models_cv[name]={}
        for mode in cv_mode:
            baseline_models_cv[name][mode] ={}
            for num_cv in cv: 
                if mode == 'y':
                    continue
                path = f'cv_models/baseline/{name}/{mode}/{num_cv}/'
                if os.path.isfile(path+'model.pkl'):
                    with open(path+'model.pkl', 'rb') as pickle_file:
                        baseline_models_cv[name][mode][num_cv] = pickle.load(pickle_file)
                else:
                    X = cv_fp_sets['train'][num_cv][mode]
                    X_val = cv_fp_sets['val'][num_cv][mode]
                    y = cv_fp_sets['train'][num_cv]['y']
                    y_val = cv_fp_sets['val'][num_cv]['y']
                    model = copy.deepcopy(models_dict[cv_model])
                    model.fit(X,y)
                    print(f'{name}_{mode}_{cv}: {model.score(X_val, y_val)}')
                    baseline_models_cv[name][mode][num_cv] = model
                    os.makedirs(path, exist_ok=True)
                    with open(path+'model.pkl', 'wb') as pickle_file:
                        pickle.dump(model,pickle_file)
                        
# baseline train-test-split training
                        
baseline_models = {}
train_name='train_split'
val_name='val_split'


if cv_model in ['RFR','PLS','MLR'] and run_cv == 'None':
    for name in cv_models:
        baseline_models[name]={}
        for mode in cv_mode:
            if mode == 'y':
                continue
            path = f'models/baseline/{name}/{mode}/'
            if os.path.isfile(path+'model.pkl'):
                with open(path+'model.pkl', 'rb') as pickle_file:
                    baseline_models[name][mode] = pickle.load(pickle_file)
            else:
                X = fp_data[train_name][mode]
                X_val = fp_data[val_name][mode]
                y = fp_data[train_name]['y']
                y_val = fp_data[val_name]['y']
                model = copy.deepcopy(models_dict[cv_model])
                model.fit(X,y)
                print(f'{name}_{mode}: {model.score(fp_data[val_name][mode], y_val)}')
                baseline_models[name][mode] = model
                os.makedirs(path, exist_ok=True)
                with open(path+'model.pkl', 'wb') as pickle_file:
                    pickle.dump(model,pickle_file)
    print(f'trained/loaded baseline models successfully')

# graph train-test-split training
if run_cv == 'None' and cv_model=='GCN' :
    graph_models = {}
    for mode in mol_modes:
        graph_models[mode] ={}
        for edge in edge_modes:
            path = f'models/gcn/{mode}/{edge}/'
            if os.path.isfile(path+'model.pkl'):
                with open(path+'model.pkl', 'rb') as pickle_file:
                    graph_models[mode][edge] = pickle.load(pickle_file)
                model = graph_models[mode][edge]
            else:
                ef = edge == 'edge'
                if mode == 'pair':
                    model = GCN_paired(edge_conv=ef).to(device=DEVICE)
                elif mode == 'deprot':
                    model = GCN_deprot(edge_conv=ef).to(device=DEVICE)
                else:
                    model = GCN_prot(edge_conv=ef).to(device=DEVICE)
                graph_models[mode][edge] = model
                os.makedirs(path, exist_ok=True)
            if model.checkpoint['epoch'] < NUM_EPOCHS:   
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                try:
                    optimizer.load_state_dict(model.checkpoint['optimizer_state'])    
                except:
                    pass
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

                print(f'Training GCN_{mode} with {edge} at epoch {model.checkpoint["epoch"]}...')
                gcn_full_training(model,loaders['train_split'],loaders['val_split'], optimizer, path)

                with open(path+'model.pkl', 'wb') as pickle_file:
                    pickle.dump(model,pickle_file)

    print(f'trained/loaded gcn models successfully')
