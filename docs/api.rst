API Documentation
=================

Query / Prediction
------------------

.. autoclass:: pkasolver.query.QueryModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pkasolver.query.States
   :members:

.. autofunction:: pkasolver.query.calculate_microstate_pka_values

.. autofunction:: pkasolver.query.calculate_pka_batch

Visualization
-------------

.. autofunction:: pkasolver.query.draw_pka_map

.. autofunction:: pkasolver.query.draw_pka_reactions

Command-Line Interface
-----------------------

.. click:: pkasolver.cli:main
   :prog: pkasolver
   :nested: full

Chemistry Utilities
-------------------

.. automodule:: pkasolver.chem
   :members:
   :undoc-members:

Data / Feature Utilities
-------------------------

.. automodule:: pkasolver.data
   :members:
   :undoc-members:

ML Utilities
------------

.. automodule:: pkasolver.ml
   :members:
   :undoc-members:

Model Architecture
------------------

.. automodule:: pkasolver.ml_architecture
   :members: GINPairV1, GINpKa
   :undoc-members:
