"""
pkasolver CLI — command-line interface for pKa prediction and visualization.

Usage
-----
  pkasolver predict --smiles "CC(=O)O"
  pkasolver predict --smiles "CC(=O)O" --output results.csv
  pkasolver predict --sdf molecules.sdf --output results.csv
  pkasolver draw --smiles "CC(=O)O" --output pka_map.png
  pkasolver draw --smiles "CC(=O)O" --reactions --output reactions.png
"""

import csv
import sys
from typing import Optional

import click
from rdkit import Chem


# ── Shared options ────────────────────────────────────────────────────────


def _load_model(num_models: int, device: Optional[str]):
    """Helper: instantiate QueryModel with CLI options."""
    from pkasolver.query import QueryModel

    return QueryModel(num_models=num_models, device=device or None)


def _states_to_rows(smiles: str, states):
    """Convert a list of States to CSV-ready row dicts."""
    rows = []
    for i, s in enumerate(states):
        rows.append(
            {
                "smiles": smiles,
                "state_index": i,
                "pka": f"{s.pka:.4f}",
                "pka_stddev": f"{s.pka_stddev:.4f}",
                "reaction_center_idx": s.reaction_center_idx,
                "protonated_smiles": Chem.MolToSmiles(s.protonated_mol),
                "deprotonated_smiles": Chem.MolToSmiles(s.deprotonated_mol),
            }
        )
    return rows


def _print_table(smiles: str, states):
    """Print pKa results as a plain-text table to stdout."""
    if not states:
        click.echo(f"No ionizable groups found for: {smiles}")
        return

    click.echo(f"\nMolecule: {smiles}")
    click.echo(f"{'#':<4} {'pKa':>8} {'±stddev':>10} {'center':>8}  protonated -> deprotonated")
    click.echo("-" * 90)
    for i, s in enumerate(states):
        prot = Chem.MolToSmiles(s.protonated_mol)
        deprot = Chem.MolToSmiles(s.deprotonated_mol)
        click.echo(
            f"{i:<4} {s.pka:>8.2f} {s.pka_stddev:>10.2f} {s.reaction_center_idx:>8}  {prot} -> {deprot}"
        )


# ── CLI group ─────────────────────────────────────────────────────────────


@click.group()
@click.version_option()
def main():
    """pkasolver: predict microstate pKa values of small molecules via GNN."""
    pass


# ── predict command ───────────────────────────────────────────────────────


@main.command()
@click.option("--smiles", "-s", default=None, help="Input SMILES string.")
@click.option(
    "--sdf",
    default=None,
    type=click.Path(exists=True),
    help="Input SDF file (batch mode).",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Output CSV file. Prints table to stdout if not provided.",
)
@click.option(
    "--num-models",
    default=25,
    show_default=True,
    help="Number of ensemble models (1–50). Use 1 for fast approximate results.",
)
@click.option(
    "--device",
    default=None,
    help="Compute device: 'cpu', 'cuda', 'cuda:0', etc. Auto-detected if not set.",
)
def predict(smiles, sdf, output, num_models, device):
    """Predict microstate pKa values for one or more molecules.

    Provide either --smiles for a single molecule, or --sdf for batch prediction.
    Results are printed as a table to stdout, or written to a CSV with --output.

    \b
    Examples:
      pkasolver predict --smiles "CC(=O)O"
      pkasolver predict --smiles "CC(=O)O" --output acetic_acid.csv
      pkasolver predict --sdf my_library.sdf --output results.csv
    """
    if not smiles and not sdf:
        raise click.UsageError("Provide --smiles or --sdf.")
    if smiles and sdf:
        raise click.UsageError("Provide only one of --smiles or --sdf, not both.")

    from pkasolver.query import calculate_microstate_pka_values

    query_model = _load_model(num_models, device)
    all_rows = []

    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise click.BadParameter(f"Invalid SMILES: {smiles}")
        try:
            states = calculate_microstate_pka_values(mol, query_model=query_model)
        except Exception as exc:
            click.echo(f"Error processing '{smiles}': {exc}", err=True)
            sys.exit(1)

        if output:
            all_rows.extend(_states_to_rows(smiles, states))
        else:
            _print_table(smiles, states)

    elif sdf:
        with open(sdf, "rb") as fh:
            suppl = Chem.ForwardSDMolSupplier(fh, removeHs=True)
            mol_list = [(Chem.MolToSmiles(m) if m else None, m) for m in suppl if m is not None]

        failed = 0
        for smi, mol in mol_list:
            try:
                states = calculate_microstate_pka_values(mol, query_model=query_model)
                if output:
                    all_rows.extend(_states_to_rows(smi, states))
                else:
                    _print_table(smi, states)
            except Exception as exc:
                click.echo(f"Warning: failed for '{smi}': {exc}", err=True)
                failed += 1

        if failed:
            click.echo(f"\n{failed}/{len(mol_list)} molecules failed.", err=True)

    if output and all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(output, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        click.echo(f"Results written to: {output}")
    elif output and not all_rows:
        click.echo("No results to write.", err=True)


# ── draw command ──────────────────────────────────────────────────────────


@main.command()
@click.option("--smiles", "-s", required=True, help="Input SMILES string.")
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output image file path (.png or .svg).",
)
@click.option(
    "--reactions",
    is_flag=True,
    default=False,
    help="Draw protonation reaction pairs instead of annotated structure map.",
)
@click.option(
    "--size",
    default=450,
    show_default=True,
    help="Image width/height in pixels (for --map mode).",
)
@click.option(
    "--row-height",
    default=250,
    show_default=True,
    help="Row height in pixels (for --reactions mode).",
)
@click.option(
    "--num-models",
    default=25,
    show_default=True,
    help="Number of ensemble models (1–50).",
)
@click.option(
    "--device",
    default=None,
    help="Compute device: 'cpu', 'cuda', etc.",
)
def draw(smiles, output, reactions, size, row_height, num_models, device):
    """Draw a pKa annotation image for a molecule.

    By default draws the molecule with pKa values annotated at each ionizable
    site (pKa map). Use --reactions to draw protonated/deprotonated pairs.

    Output format is determined by the file extension (.png or .svg).
    Note: --reactions mode always writes PNG (via cairosvg).

    \b
    Examples:
      pkasolver draw --smiles "CC(=O)O" --output pka_map.png
      pkasolver draw --smiles "c1ccncc1" --reactions --output pyridine_rxn.png
    """
    from pkasolver.query import (
        QueryModel,
        calculate_microstate_pka_values,
        draw_pka_map,
        draw_pka_reactions,
    )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise click.BadParameter(f"Invalid SMILES: {smiles}")

    query_model = _load_model(num_models, device)

    try:
        states = calculate_microstate_pka_values(mol, query_model=query_model)
    except Exception as exc:
        raise click.ClickException(f"pKa prediction failed: {exc}")

    if not states:
        raise click.ClickException("No ionizable groups found.")

    if reactions:
        # draw_pka_reactions writes PNG internally via cairosvg
        draw_pka_reactions(states, height=row_height, write_png_to_file=output)
        click.echo(f"Reaction diagram written to: {output}")
    else:
        img = draw_pka_map(states, size=(size, size))
        ext = output.lower().rsplit(".", 1)[-1] if "." in output else "png"
        if ext == "png":
            img.save(output)
        elif ext == "svg":
            # RDKit can render to SVG directly
            from rdkit.Chem.Draw import rdMolDraw2D

            drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
            drawer.DrawMolecule(states[0].ph7_mol)
            drawer.FinishDrawing()
            with open(output, "w") as fh:
                fh.write(drawer.GetDrawingText())
        else:
            img.save(output)
        click.echo(f"pKa map written to: {output}")
