pkasolver
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/mayrf/pkasolver/workflows/CI/badge.svg)](https://github.com/mayrf/pkasolver/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/mayrf/pkasolver/branch/master/graph/badge.svg)](https://codecov.io/gh/mayrf/pkasolver/branch/master)

pKasolver is a package that enables prediction of pKa values of small molecules via graph convolutional networks. 
This repository and data was used for the preprint [**Improving Small Molecule pKa Prediction Using Transfer Learning with Graph Neural Networks**](https://www.biorxiv.org/content/10.1101/2022.01.20.476787v1).

Due to the restrictive licence of Schrödinger's Epik, we are unfortunatly unable to distribute the models that were trained using the transfer learning approach (pkasolver-epic), described by this publication. Instead, the model provided with this repository and used by the Google Colab Jupyter notebook, described below, has been trained without the transfer learning step (pkasolver-lite). For details about the limitations of this models see the publication. 


<!-- ## Prerequisites -->

### Installation

We recommend using anaconda to set up a new repository (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) and installing the dependencies listed in `devtools/conda-envs/test_env.yaml`.
This can be done with `conda env create -f devtools/conda-envs/test_env.yaml` (the conda environment will be called `test` --- if you want a different name change the first line in `test_env.yaml`).
After activating the conda environment the package can be installed with `python setup.py install`.

## How to use pkasolver to predict microstate pka values

Depending on your needs you can either use the juptyer notebook deposited in the `notebooks` directory which demonstrate the usage of the pkasolver package (this needs a local installation of the pkasolver package).
Or you can use the provided google colabs that automatically installs everything in a hosted notebook and allows you to use the package for pka prediction without a local installation: https://github.com/mayrf/pkasolver/blob/main/notebooks/pka_prediction.ipynb

To generate your own models, take a look at the `scripts` folder in the corresponding data repository: https://github.com/wiederm/pkasolver-data. It contains notebooks to reproduce the plots shwon in [placeholder] and python scripts to prepare the data, train models and visualize their performance. Note that an active license for Schrödinger's Epik is required in order to reproduce the transfer learning dataset.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

### Copyright

Copyright (c) 2022, Fritz Mayr, Oliver Wieder, Marcus Wieder, Thierry Langer


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.4.
