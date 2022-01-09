#pkasolver
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/MayrF/pkasolver/workflows/CI/badge.svg)](https://github.com/MayrF/pkasolver/actions?query=workflow%3ACI)
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

1) Download this repo
2) Install all dependencies in your python environment. (yaml file is located in devtools/conda-envs/test_env.yml) (Note that the dependencies may take a long time to download)
3) run `conda activate pkasolver` to activate conda environment

option a) pip install git+https://github.com/mayrf/pkasolver.git
option b) `cd` into the repo and install the pkasolver package by using the terminal command `python setup.py install`

## Usage

The `notebooks` directory contains jupyter notebooks that demonstrate the usage of the pkasolver api for predicting pka values for single or multiple molecules. The model used in the api is included with the package files.
To generate your own models, see the `scripts` folder which contains a number of python scripts to prepare data, train models and benchmark their performance.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

### Copyright

Copyright (c) 2021, Fritz Mayr, Marcus Wieder, Oliver Wieder


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.4.
This project has been inspired by the works of Baltruschat et. al. (https://f1000research.com/articles/9-113/v2) and Pan et. al. (https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00075) and was developed in the course of the master thesis of Fritz Mayr at the group of Dr. Thierry Langer at the Department of Pharmaceutical Sciences of the University of Vienna.

