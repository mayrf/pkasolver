Getting Started
===============

Installation
------------

**uv (recommended):**

.. code-block:: bash

   pip install uv
   uv pip install pkasolver

**pip:**

.. code-block:: bash

   pip install pkasolver

**conda:**

.. code-block:: bash

   git clone --depth 1 https://github.com/mayrf/pkasolver
   cd pkasolver
   conda env create -f devtools/conda-envs/test_env.yaml
   conda activate pka_prediction
   pip install -e .

**GPU (CUDA):**

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install pkasolver

**GPU (ROCm/AMD):**

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
   pip install pkasolver

**Nix:**

.. code-block:: bash

   nix develop          # CPU
   nix develop .#cuda   # CUDA
   nix develop .#rocm   # ROCm

.. note::

   For large-scale usage, clone with ``--depth 1`` to avoid downloading the
   full git history (which contains legacy large binary files)::

       git clone --depth 1 https://github.com/mayrf/pkasolver


Use Cases
---------

1. Single Molecule Prediction (Python API)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rdkit import Chem
   from pkasolver.query import QueryModel, calculate_microstate_pka_values

   model = QueryModel()  # loads 25-model ensemble
   mol = Chem.MolFromSmiles("CC(=O)O")
   states = calculate_microstate_pka_values(mol, query_model=model)

   for state in states:
       print(f"pKa = {state.pka:.2f} ± {state.pka_stddev:.2f}")

Each element of ``states`` is a :class:`pkasolver.query.States` dataclass with fields:

- ``pka`` — predicted pKa value
- ``pka_stddev`` — ensemble standard deviation (uncertainty estimate)
- ``protonated_mol`` — RDKit Mol of the protonated form
- ``deprotonated_mol`` — RDKit Mol of the deprotonated form
- ``reaction_center_idx`` — atom index of the ionization site
- ``ph7_mol`` — RDKit Mol at pH 7

**Fewer models for faster inference:**

.. code-block:: python

   model = QueryModel(num_models=1)  # fast, less accurate

2. Batch Prediction (Python API)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pkasolver.query import QueryModel, calculate_pka_batch

   model = QueryModel()
   smiles_list = ["CC(=O)O", "c1ccncc1", "C1CCNCC1"]
   results = calculate_pka_batch(smiles_list, query_model=model)

   for smi, states in zip(smiles_list, results):
       if states is None:
           print(f"{smi}: failed")
       else:
           print(f"{smi}: {[round(s.pka, 2) for s in states]}")

:func:`pkasolver.query.calculate_pka_batch` loads the model once and handles
per-molecule failures gracefully — it logs errors and returns ``None`` for
failed molecules rather than aborting the batch.

3. Visualization
~~~~~~~~~~~~~~~~

.. code-block:: python

   from pkasolver.query import draw_pka_map, draw_pka_reactions

   # Annotated structure at pH 7
   img = draw_pka_map(states, size=(600, 600))
   img.save("pka_map.png")

   # Reaction diagram (protonated ↔ deprotonated grid)
   svg = draw_pka_reactions(states, height=300, write_png_to_file="reactions.png")

4. Command-Line Interface
~~~~~~~~~~~~~~~~~~~~~~~~~

**Predict from SMILES:**

.. code-block:: bash

   pkasolver predict --smiles "CC(=O)O"

**Batch predict from SDF to CSV:**

.. code-block:: bash

   pkasolver predict --sdf library.sdf --output results.csv

**Draw pKa map:**

.. code-block:: bash

   pkasolver draw --smiles "CC(=O)O" --output pka_map.png

**Draw reaction pairs:**

.. code-block:: bash

   pkasolver draw --smiles "c1ccncc1" --reactions --output reactions.png

**GPU inference:**

.. code-block:: bash

   pkasolver predict --smiles "CC(=O)O" --device cuda

5. Jupyter Notebooks
~~~~~~~~~~~~~~~~~~~~

``notebooks/pka_prediction.ipynb``
   Google Colab-compatible notebook. No local installation needed.
   Open in Colab: https://colab.research.google.com/github/mayrf/pkasolver/blob/main/notebooks/pka_prediction.ipynb

``notebooks/query_example.ipynb``
   Local usage examples: QueryModel API, visualization, batch processing.
   Requires a local installation.

6. Training Your Own Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training scripts and evaluation notebooks live in a separate repository:
https://github.com/wiederm/pkasolver-data

That repository contains:

- Data preparation scripts (SDF → PyG dataset)
- Model training scripts
- Evaluation notebooks reproducing publication plots
- Notes on the Epik transfer learning dataset (requires Schrödinger licence)
