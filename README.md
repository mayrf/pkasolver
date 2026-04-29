# pkasolver

[![CI](https://github.com/mayrf/pkasolver/workflows/CI/badge.svg)](https://github.com/mayrf/pkasolver/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/mayrf/pkasolver/branch/main/graph/badge.svg)](https://codecov.io/gh/mayrf/pkasolver/branch/main)
[![PyPI version](https://img.shields.io/pypi/v/pkasolver.svg)](https://pypi.org/project/pkasolver/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

**pkasolver** predicts microstate pKa values of small molecules using Graph Isomorphism Networks (GINs). It provides a Python API, a command-line interface, and Jupyter notebooks.

Based on the publication: [**Improving Small Molecule pKa Prediction Using Transfer Learning with Graph Neural Networks**](https://www.biorxiv.org/content/10.1101/2022.01.20.476787v1) — Mayr et al., 2022.

> **Note on models:** Due to the restrictive licence of Schrödinger's Epik, the transfer-learned model (pkasolver-epik) cannot be distributed. The model shipped with this package (**pkasolver-lite**) was trained without the Epik transfer learning step. See the publication for a discussion of the accuracy differences.

---

## Table of Contents

- [Installation](#installation)
  - [uv (recommended)](#uv-recommended)
  - [pip](#pip)
  - [conda](#conda)
  - [GPU support (CUDA / ROCm)](#gpu-support-cuda--rocm)
  - [Nix](#nix)
- [Quick Start](#quick-start)
- [Use Cases](#use-cases)
  - [1. Single molecule prediction (Python API)](#1-single-molecule-prediction-python-api)
  - [2. Batch prediction (Python API)](#2-batch-prediction-python-api)
  - [3. Visualization](#3-visualization)
  - [4. Command-line interface](#4-command-line-interface)
  - [5. Notebooks](#5-notebooks)
  - [6. Training your own models](#6-training-your-own-models)
- [License](#license)

---

## Installation

> **Tip:** For large-scale usage, clone with `--depth 1` to avoid downloading the full git history (which contains legacy large files):
> ```bash
> git clone --depth 1 https://github.com/mayrf/pkasolver
> ```

### uv (recommended)

[uv](https://github.com/astral-sh/uv) is the fastest way to get started:

```bash
pip install uv            # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install pkasolver
```

Or install from source:

```bash
git clone --depth 1 https://github.com/mayrf/pkasolver
cd pkasolver
uv pip install -e ".[dev]"
```

### pip

```bash
pip install pkasolver
```

From source:

```bash
pip install -e ".[dev]"
```

### conda

```bash
git clone --depth 1 https://github.com/mayrf/pkasolver
cd pkasolver
conda env create -f devtools/conda-envs/test_env.yaml
conda activate pka_prediction
pip install -e .
```

### GPU support (CUDA / ROCm)

pkasolver automatically uses a GPU if one is available. PyTorch supports both
CUDA (NVIDIA) and ROCm (AMD) through the same `torch.cuda.is_available()` API.

**CUDA:**
```bash
# Install PyTorch with CUDA 12.1 support, then install pkasolver
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install pkasolver
```

**ROCm:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
pip install pkasolver
```

To force CPU inference regardless of available hardware:
```python
from pkasolver.query import QueryModel
model = QueryModel(device="cpu")
```

### Nix

A `flake.nix` is provided with CPU, CUDA, and ROCm development shells:

```bash
nix develop                  # CPU dev shell (default)
nix develop .#cuda           # CUDA dev shell
nix develop .#rocm           # ROCm dev shell
```

---

## Quick Start

```python
from rdkit import Chem
from pkasolver.query import calculate_microstate_pka_values, draw_pka_map

mol = Chem.MolFromSmiles("CC(=O)O")  # acetic acid
states = calculate_microstate_pka_values(mol)

for state in states:
    print(f"pKa = {state.pka:.2f} ± {state.pka_stddev:.2f}")

# Draw annotated structure (returns PIL Image in scripts, displays in notebooks)
img = draw_pka_map(states)
img.save("pka_map.png")
```

---

## Use Cases

### 1. Single molecule prediction (Python API)

```python
from rdkit import Chem
from pkasolver.query import QueryModel, calculate_microstate_pka_values

# Load model once (25-model ensemble by default)
model = QueryModel()

mol = Chem.MolFromSmiles("c1ccncc1")  # pyridine
states = calculate_microstate_pka_values(mol, query_model=model)

for state in states:
    print(f"  pKa = {state.pka:.2f} ± {state.pka_stddev:.2f}")
    print(f"  Protonated:   {Chem.MolToSmiles(state.protonated_mol)}")
    print(f"  Deprotonated: {Chem.MolToSmiles(state.deprotonated_mol)}")
    print(f"  Site index:   {state.reaction_center_idx}")
```

**Faster inference with fewer models:**
```python
# Use 1 model for quick screening (less accurate, much faster)
model = QueryModel(num_models=1)
```

### 2. Batch prediction (Python API)

For processing many molecules, use `calculate_pka_batch` which loads the model once
and handles per-molecule errors gracefully (no silent hangs):

```python
from pkasolver.query import QueryModel, calculate_pka_batch

smiles_list = [
    "CC(=O)O",    # acetic acid
    "c1ccncc1",   # pyridine
    "C1CCNCC1",   # piperidine
    "Cc1ccccc1",  # toluene (no ionizable group)
]

model = QueryModel()
results = calculate_pka_batch(smiles_list, query_model=model)

for smi, states in zip(smiles_list, results):
    if states is None:
        print(f"{smi}: failed")
    elif len(states) == 0:
        print(f"{smi}: no ionizable groups")
    else:
        pkas = [f"{s.pka:.2f}" for s in states]
        print(f"{smi}: pKa = {', '.join(pkas)}")
```

**From an SDF file:**
```python
from rdkit import Chem
from pkasolver.query import QueryModel, calculate_microstate_pka_values

model = QueryModel()
with open("library.sdf", "rb") as fh:
    suppl = Chem.ForwardSDMolSupplier(fh, removeHs=True)
    for mol in suppl:
        if mol is None:
            continue
        states = calculate_microstate_pka_values(mol, query_model=model)
        for s in states:
            print(Chem.MolToSmiles(mol), s.pka)
```

### 3. Visualization

**pKa map** — annotated structure at pH 7 with pKa values at each ionizable site:

```python
from pkasolver.query import draw_pka_map
img = draw_pka_map(states, size=(600, 600))
img.save("pka_map.png")
# In Jupyter: display(img)
```

**Reaction diagram** — grid of protonated/deprotonated pairs with equilibrium arrows:

```python
from pkasolver.query import draw_pka_reactions
svg = draw_pka_reactions(states, height=300, write_png_to_file="reactions.png")
# In Jupyter: display(svg)
```

### 4. Command-line interface

**Predict from SMILES (stdout table):**
```bash
pkasolver predict --smiles "CC(=O)O"
```

**Predict to CSV:**
```bash
pkasolver predict --smiles "c1ccncc1" --output pyridine.csv
```

**Batch predict from SDF:**
```bash
pkasolver predict --sdf my_library.sdf --output results.csv
```

**Fast mode (single model):**
```bash
pkasolver predict --smiles "CC(=O)O" --num-models 1
```

**Draw pKa map:**
```bash
pkasolver draw --smiles "CC(=O)O" --output pka_map.png
```

**Draw reaction pairs:**
```bash
pkasolver draw --smiles "c1ccncc1" --reactions --output pyridine_reactions.png
```

**GPU inference:**
```bash
pkasolver predict --smiles "CC(=O)O" --device cuda
```

### 5. Notebooks

| Notebook | Description | Launch |
|---|---|---|
| `notebooks/pka_prediction.ipynb` | pKa prediction — Google Colab compatible, no local install needed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mayrf/pkasolver/blob/main/notebooks/pka_prediction.ipynb) |
| `notebooks/query_example.ipynb` | Local usage: QueryModel API, visualization, batch processing | Local only |

### 6. Training your own models

Training scripts, data preparation, and evaluation notebooks live in a separate repository:

**[github.com/wiederm/pkasolver-data](https://github.com/wiederm/pkasolver-data)**

That repository contains:
- Python scripts to prepare training data (SDF → PyG dataset)
- Model training scripts with hyperparameter configuration
- Evaluation notebooks reproducing the plots from the publication
- Notes on reproducing the Epik transfer learning dataset (requires Schrödinger licence)

---

## License

This project is licensed under the MIT License — see [LICENSE.md](LICENSE.md) for details.

### Copyright

Copyright (c) 2022, Fritz Mayr, Oliver Wieder, Marcus Wieder, Thierry Langer

### Acknowledgements

Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.4.
