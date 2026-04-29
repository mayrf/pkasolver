"""
Tests for the pkasolver CLI.

Uses click's CliRunner for isolated, subprocess-free invocation.
Model tests run with --num-models 1 for speed in CI.
"""
import csv
import os

import pytest
from click.testing import CliRunner

from pkasolver.cli import main


@pytest.fixture()
def runner():
    return CliRunner()


# ── predict command ───────────────────────────────────────────────────────


def test_predict_stdout(runner):
    """predict with --smiles prints a table to stdout."""
    result = runner.invoke(main, ["predict", "--smiles", "CC(=O)O", "--num-models", "1"])
    assert result.exit_code == 0, result.output
    assert "pKa" in result.output or "pka" in result.output.lower()


def test_predict_csv_output(runner, tmp_path):
    """predict with --output writes a valid CSV."""
    out = str(tmp_path / "results.csv")
    result = runner.invoke(
        main,
        ["predict", "--smiles", "CC(=O)O", "--num-models", "1", "--output", out],
    )
    assert result.exit_code == 0, result.output
    assert os.path.exists(out)
    with open(out) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) >= 1
    assert "pka" in rows[0]
    assert "smiles" in rows[0]


def test_predict_invalid_smiles(runner):
    """predict exits non-zero for invalid SMILES."""
    result = runner.invoke(main, ["predict", "--smiles", "not_a_smiles", "--num-models", "1"])
    assert result.exit_code != 0


def test_predict_no_input(runner):
    """predict exits with UsageError if no --smiles or --sdf given."""
    result = runner.invoke(main, ["predict"])
    assert result.exit_code != 0
    assert "smiles" in result.output.lower() or "sdf" in result.output.lower()


def test_predict_sdf(runner, tmp_path):
    """predict accepts an SDF file and writes results."""
    from rdkit import Chem
    from rdkit.Chem import SDWriter

    sdf_path = str(tmp_path / "test.sdf")
    out_path = str(tmp_path / "results.csv")

    # Write a simple SDF with two molecules
    with SDWriter(sdf_path) as w:
        w.write(Chem.MolFromSmiles("CC(=O)O"))
        w.write(Chem.MolFromSmiles("c1ccncc1"))

    result = runner.invoke(
        main,
        ["predict", "--sdf", sdf_path, "--num-models", "1", "--output", out_path],
    )
    assert result.exit_code == 0, result.output
    assert os.path.exists(out_path)
    with open(out_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 2


# ── draw command ──────────────────────────────────────────────────────────


def test_draw_map_png(runner, tmp_path):
    """draw produces a PNG file."""
    out = str(tmp_path / "pka_map.png")
    result = runner.invoke(
        main,
        ["draw", "--smiles", "CC(=O)O", "--output", out, "--num-models", "1"],
    )
    assert result.exit_code == 0, result.output
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_draw_requires_output(runner):
    """draw exits with error if --output not given."""
    result = runner.invoke(main, ["draw", "--smiles", "CC(=O)O"])
    assert result.exit_code != 0


def test_draw_invalid_smiles(runner, tmp_path):
    """draw exits non-zero for invalid SMILES."""
    out = str(tmp_path / "pka_map.png")
    result = runner.invoke(
        main,
        ["draw", "--smiles", "not_a_smiles", "--output", out, "--num-models", "1"],
    )
    assert result.exit_code != 0
