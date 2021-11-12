#pkasolver
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/MayrF/pkasolver.svg?branch=master)](https://travis-ci.com/MayrF/pkasolver)
[![codecov](https://codecov.io/gh/MayrF/pkasolver/branch/master/graph/badge.svg)](https://codecov.io/gh/MayrF/pkasolver/branch/master)

pKasolver is a package that enables prediction of pKa values of small molecules via graph convolutional networks. 

## Prerequisites

The Python dependencies are:
* Python >= 3.7
* NumPy >= 1.18
* Scikit-Learn >= 0.22
* PyTorch >= 1.9
* PyTorch Geometric >= 1.7.2 
* RDKit >= 2019.09.3
* Pandas >= 0.25
* JupyterLab >= 1.2
* Matplotlib >= 3.1
* Seaborn >= 0.9

### Installing

1) Install all dependencies in your python environment. 
2) Download this repo
3) `cd` into the repo
3) Install the package by using the terminal command `python setup.py install`

## Usage

The usage of this package is demonstrated by files in the `examples/tesis_run` folder. The juypter notebook `thesis_pipeline.ipynb` that contains the entire workflow of data preprocessing, featurization, model training and testing, except for the cross-validation training.The script `train_script.py` enables parallel computation of all train-test-split and cross-validation models on multiple machines. This can be done by using the script in conjuction with the files test.sh and `run_training.sh`. The global variables are stored and called from the file `config.py` and the code for the neural network architectures as well as the training functions from the file `architecture.py`. 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

It uses pyTorch and pyTorch geometrics for handling NN-models and sklearn to deploy other machine learning models that serve as baselines.  for  is inspired by the works of Baltruschat et. al. and 

developed in the course of the master tesis of Fritz Mayr at the 

toolset for predicting the pka values of small molecules

### Copyright

Copyright (c) 2020, Fritz Mayr


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.4.
