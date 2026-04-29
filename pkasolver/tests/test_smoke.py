"""
Smoke tests for core pKa prediction functionality.

These tests run in CI (using a single-model ensemble for speed) and verify
that the key pipeline — dimorphite-dl direct import, model loading, and
pKa prediction — works end-to-end.

Issue coverage:
  #34 - KeyError in model state_dict (PyG version mismatch)
  #37 - subprocess used for dimorphite_dl (replaced with direct import)
  #35 - silent hang on large libraries (fixed by direct import + batch helper)
"""
import pytest
from rdkit import Chem


# ── Model loading (issue #34) ──────────────────────────────────────────────


def test_query_model_loads(query_model):
    """QueryModel loads without KeyError regardless of PyG version."""
    assert len(query_model.models) >= 1


def test_query_model_custom_num_models():
    """QueryModel can be instantiated with a custom number of models."""
    from pkasolver.query import QueryModel

    qm = QueryModel(num_models=1)
    assert len(qm.models) == 1


# ── dimorphite_dl direct import (issues #37, #35) ──────────────────────────


def test_dimorphite_dl_direct_import():
    """dimorphite_dl can be called directly (no subprocess)."""
    from pkasolver.query import _call_dimorphite_dl

    mol = Chem.MolFromSmiles("CC(=O)O")  # acetic acid
    result = _call_dimorphite_dl(mol, min_ph=0.5, max_ph=13.5)
    assert isinstance(result, list)
    assert len(result) >= 1
    assert all(isinstance(m, Chem.rdchem.Mol) for m in result)


def test_dimorphite_dl_ph7():
    """dimorphite_dl returns single mol at pH 7 for acetic acid."""
    from pkasolver.query import _call_dimorphite_dl

    mol = Chem.MolFromSmiles("CC(=O)O")
    result = _call_dimorphite_dl(mol, min_ph=7.0, max_ph=7.0, pka_precision=0)
    assert len(result) == 1


# ── End-to-end pKa prediction ─────────────────────────────────────────────


def test_pka_acetic_acid(query_model):
    """Acetic acid pKa is predicted in a reasonable range (~4.75)."""
    from pkasolver.query import calculate_microstate_pka_values

    mol = Chem.MolFromSmiles("CC(=O)O")
    states = calculate_microstate_pka_values(mol, query_model=query_model)
    assert len(states) >= 1
    pkas = [s.pka for s in states]
    # Acetic acid pKa ~ 4.75; allow wide tolerance for single-model subset
    assert any(2.0 < p < 8.0 for p in pkas), f"No pKa in [2,8] range, got: {pkas}"


def test_pka_piperidine(query_model):
    """Piperidine pKa is predicted in a reasonable range (~10.5 for conjugate acid)."""
    from pkasolver.query import calculate_microstate_pka_values

    mol = Chem.MolFromSmiles("C1CCNCC1")
    states = calculate_microstate_pka_values(mol, query_model=query_model)
    assert len(states) >= 1
    pkas = [s.pka for s in states]
    assert any(7.0 < p < 13.0 for p in pkas), f"No pKa in [7,13] range, got: {pkas}"


def test_pka_returns_states_dataclass(query_model):
    """calculate_microstate_pka_values returns States objects with expected fields."""
    from pkasolver.query import States, calculate_microstate_pka_values

    mol = Chem.MolFromSmiles("Nc1ccccc1")  # aniline
    states = calculate_microstate_pka_values(mol, query_model=query_model)
    for state in states:
        assert isinstance(state, States)
        assert isinstance(state.pka, float)
        assert isinstance(state.pka_stddev, float)
        assert state.protonated_mol is not None
        assert state.deprotonated_mol is not None
        assert isinstance(state.reaction_center_idx, int)


# ── Batch API (issue #35) ──────────────────────────────────────────────────


def test_batch_prediction(query_model):
    """calculate_pka_batch processes multiple molecules without hanging."""
    from pkasolver.query import calculate_pka_batch

    smiles_list = [
        "CC(=O)O",       # acetic acid
        "C1CCNCC1",      # piperidine
        "c1ccncc1",      # pyridine
    ]
    results = calculate_pka_batch(smiles_list, query_model=query_model)
    assert len(results) == 3
    # All should succeed (not None)
    for i, res in enumerate(results):
        assert res is not None, f"Molecule {i} ({smiles_list[i]}) returned None"
        assert len(res) >= 1


def test_batch_prediction_handles_invalid_smiles(query_model):
    """calculate_pka_batch skips invalid SMILES and returns None for them."""
    from pkasolver.query import calculate_pka_batch

    smiles_list = ["CC(=O)O", "not_a_smiles", "C1CCNCC1"]
    results = calculate_pka_batch(smiles_list, query_model=query_model)
    assert len(results) == 3
    assert results[0] is not None
    assert results[1] is None
    assert results[2] is not None


# ── Device configurability ────────────────────────────────────────────────


def test_query_model_cpu_device():
    """QueryModel can be explicitly set to CPU."""
    from pkasolver.query import QueryModel

    qm = QueryModel(num_models=1, device="cpu")
    assert str(qm.device) == "cpu"
    # Verify models are on CPU
    for model in qm.models:
        for param in model.parameters():
            assert param.device.type == "cpu"
            break
