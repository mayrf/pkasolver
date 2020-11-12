import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem


import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats


def importsdf(sdf):
    df = PandasTools.LoadSDF(sdf)
    df["smiles"] = [Chem.MolToSmiles(m) for m in df["ROMol"]]
    return df

def morganfp(df,neigh,nB,useFeatures=True):
    #create morganfp and ad to df
    df['morganfp']= [AllChem.GetMorganFingerprintAsBitVect(m,neigh,nBits=nB,useFeatures=useFeatures) for m in df["ROMol"]] 
    return df

def make_fp_array(df):
    return np.array([np.array(row) for row in df['morganfp']])

def make_stat_variables(df, X_list, y_name):
    X = np.asfarray(df[X_list],float)
    y = np.asfarray(df[y_name],float).reshape(-1)
    return X, y

def catVariables(X_feat,X_fp):
    return np.concatenate((X_feat,X_fp),axis=1)

def plotresults(prediction,true_vals,name):
    a = {'y': true_vals, 'y_hat': prediction}
    df = pd.DataFrame(data=a)

    fig= plt.figure(figsize=(18,13))
    fig.suptitle(name)
    plt.subplot(221)
    
    r2=(stats.pearsonr(df['y'],df['y_hat'])[0])**2
    
    # calculate manually
    d = df['y'] - df['y_hat']
    mse_f = np.mean(d**2)
    mae_f = np.mean(abs(d))
    rmse_f = np.sqrt(mse_f)
    r2_f = 1-(sum(d**2)/sum((df['y']-np.mean(df['y']))**2))
    
    stat_info = f"""
    MAE = {mae_f:.2}
    MASE = {mae_f:.2}
    RMSE = {rmse_f:.2}
    $r^2$ = {r2_f:.2} 
    """
    
    ax = sns.regplot(x='y', y='y_hat', data=df)
    ax.text(0, 1, stat_info, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, linespacing=1)
    ax.set_xlabel('$\mathrm{pKa}_{exp}$')
    ax.set_ylabel('$\mathrm{pKa}_{calc}$')

    plt.subplot(222)
    ax = sns.distplot(df['y'],bins=20, label='$\mathrm{pKa}_{exp}$')
    sns.distplot(df['y_hat'],bins=20, label='$\mathrm{pKa}_{calc}$')
    ax.set_xlabel('pKa')
    ax.legend()

    plt.subplot(223)
    ax = sns.distplot(df['y']-df['y_hat'], bins=10)
    ax.set_xlabel('$\mathrm{pKa}_{exp} - \mathrm{pKa}_{calc}$')
    
    plt.show()
    plt.close()


