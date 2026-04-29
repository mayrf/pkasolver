# imports
import logging
from copy import deepcopy
from dataclasses import dataclass
from operator import attrgetter
from os import path
from typing import List, Optional, Union

import cairosvg
import numpy as np
import svgutils.transform as sg
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from torch_geometric.loader import DataLoader

from pkasolver.chem import create_conjugate
from pkasolver.constants import DEVICE, EDGE_FEATURES, NODE_FEATURES, SEED
from pkasolver.data import (
    calculate_nr_of_features,
    make_features_dicts,
    mol_to_paired_mol_data,
)
from pkasolver.ml import dataset_to_dataloader
from pkasolver.ml_architecture import GINPairV1


@dataclass
class States:
    pka: float
    pka_stddev: float
    protonated_mol: Chem.Mol
    deprotonated_mol: Chem.Mol
    reaction_center_idx: int
    ph7_mol: Chem.Mol


logger = logging.getLogger(__name__)

RDLogger.DisableLog("rdApp.*")

node_feat_list = [
    "element",
    "formal_charge",
    "hybridization",
    "total_num_Hs",
    "aromatic_tag",
    "total_valence",
    "total_degree",
    "is_in_ring",
    "reaction_center",
    "smarts",
]

edge_feat_list = ["bond_type", "is_conjugated", "rotatable"]
num_node_features = calculate_nr_of_features(node_feat_list)
num_edge_features = calculate_nr_of_features(edge_feat_list)

# make dicts from selection list to be used in the processing step
selected_node_features = make_features_dicts(NODE_FEATURES, node_feat_list)
selected_edge_features = make_features_dicts(EDGE_FEATURES, edge_feat_list)


class QueryModel:
    """Loads an ensemble of pre-trained GINPairV1 models and predicts pKa values.

    Parameters
    ----------
    num_models : int
        Number of ensemble models to load (max 50, default 25).
        Use a lower value (e.g. 1) for faster inference at the cost of accuracy.
    device : torch.device or str, optional
        Device to run inference on. Defaults to the global DEVICE (auto-detected).
    """

    def __init__(
        self,
        num_models: int = 25,
        device: Optional[Union[torch.device, str]] = None,
    ):
        self.device = torch.device(device) if device is not None else DEVICE
        self.models = []

        base_path = path.dirname(__file__)
        for i in range(num_models):
            model = GINPairV1(num_node_features, num_edge_features, hidden_channels=96)
            checkpoint = torch.load(
                f"{base_path}/trained_model_without_epik/best_model_{i}.pt",
                map_location=self.device,
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            model.to(device=self.device)
            self.models.append(model)

    def predict_pka_value(self, loader: DataLoader) -> np.ndarray:
        """Predict pKa for a single ionization event.

        Parameters
        ----------
        loader : DataLoader
            DataLoader containing exactly one PairData graph.

        Returns
        -------
        tuple[float, float]
            (mean_pka, stddev_pka) from the ensemble.
        """
        results = []
        assert len(loader) == 1
        for data in loader:
            data.to(device=self.device)
            consensus_r = []
            for model in self.models:
                y_pred = (
                    model(
                        x_p=data.x_p,
                        x_d=data.x_d,
                        edge_attr_p=data.edge_attr_p,
                        edge_attr_d=data.edge_attr_d,
                        data=data,
                    )
                    .reshape(-1)
                    .detach()
                )
                consensus_r.append(y_pred.tolist())
            results.extend(
                (
                    float(np.average(consensus_r, axis=0)),
                    float(np.std(consensus_r, axis=0)),
                )
            )
        return results


def _get_ionization_indices(mol_list: list, compare_to: Chem.Mol) -> list:
    """Takes a list of mol objects of different protonation states,
    and returns the protonation center index.
    """
    from rdkit.Chem import rdFMCS

    list_of_reaction_centers = []
    for idx, m2 in enumerate(mol_list):
        m1 = compare_to
        assert m1.GetNumAtoms() == m2.GetNumAtoms()

        mcs = rdFMCS.FindMCS(
            [m1, m2],
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            timeout=120,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
        )

        mcsp = Chem.MolFromSmarts(mcs.smartsString, False)
        s1 = m1.GetSubstructMatch(mcsp)
        s2 = m2.GetSubstructMatch(mcsp)

        for i, j in zip(s1, s2):
            if i != j:
                break
            if m1.GetAtomWithIdx(i).GetFormalCharge() != m2.GetAtomWithIdx(j).GetFormalCharge():
                if i != j:
                    logger.warning("Trouble ahead ... different atom indices detected.")
                list_of_reaction_centers.append(i)

    logger.debug(set(list_of_reaction_centers))
    return list_of_reaction_centers


def _call_dimorphite_dl(
    mol: Chem.Mol,
    min_ph: float,
    max_ph: float,
    pka_precision: float = 1.0,
) -> List[Chem.Mol]:
    """Enumerate protonation states using dimorphite-dl (direct Python call).

    Replaces the old subprocess approach which caused hangs on large libraries
    (issue #35) and was unnecessarily slow (issue #37).
    """
    from pkasolver.dimorphite_dl.dimorphite_dl import run_with_mol_list

    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    # run_with_mol_list accepts the same kwargs as the CLI arguments
    result_mols = run_with_mol_list(
        [mol],
        min_ph=min_ph,
        max_ph=max_ph,
        pka_precision=pka_precision,
        silent=True,
        max_variants=128,
    )
    return result_mols


def _sort_conj(mols: list):
    """Sort mols based on number of hydrogens (protonated first)."""
    assert len(mols) == 2
    nr_of_hydrogen = [np.sum([atom.GetTotalNumHs() for atom in mol.GetAtoms()]) for mol in mols]
    if abs(nr_of_hydrogen[0] - nr_of_hydrogen[1]) != 1:
        raise RuntimeError(
            "Neighboring protonation states are only allowed to have a difference of a single hydrogen."
        )
    mols_sorted = [x for _, x in sorted(zip(nr_of_hydrogen, mols), reverse=True)]
    return mols_sorted


def _check_for_duplicates(states: list):
    """Remove duplicate protonation states with identical pKa values."""
    all_r = dict()
    logger.debug(states)
    for state in states:
        m1, m2 = _sort_conj([state.protonated_mol, state.deprotonated_mol])
        all_r[hash((Chem.MolToSmiles(m1), Chem.MolToSmiles(m2)))] = state
    return sorted([all_r[k] for k in all_r], key=attrgetter("pka"))


def calculate_microstate_pka_values(
    mol: Chem.rdchem.Mol,
    only_dimorphite: bool = False,
    query_model: Optional[QueryModel] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> List[States]:
    """Predict microstate pKa values for a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object.
    only_dimorphite : bool
        If True, use only dimorphite-dl for protonation state enumeration
        (experimental, may produce incorrect states).
    query_model : QueryModel, optional
        Pre-loaded QueryModel instance. If None, a new one is created.
        Pass an existing instance when processing many molecules to avoid
        reloading the ensemble for each call.
    device : torch.device or str, optional
        Compute device override. Defaults to auto-detected DEVICE.

    Returns
    -------
    list[States]
        List of States dataclass objects sorted by pKa value, each containing
        pka, pka_stddev, protonated_mol, deprotonated_mol, reaction_center_idx, ph7_mol.
    """
    if query_model is None:
        query_model = QueryModel(device=device)

    if only_dimorphite:
        logger.warning(
            "only_dimorphite=True is experimental and might generate wrong protonation states."
        )
        logger.debug("Using dimorphite-dl to enumerate protonation states.")
        mol_at_ph_7 = _call_dimorphite_dl(mol, min_ph=7.0, max_ph=7.0, pka_precision=0)
        all_mols = _call_dimorphite_dl(mol, min_ph=0.5, max_ph=13.5)
        atom_charges = [np.sum([atom.GetTotalNumHs() for atom in mol.GetAtoms()]) for mol in all_mols]
        mols_sorted = [x for _, x in sorted(zip(atom_charges, all_mols), reverse=True)]

        reaction_center_atom_idxs = _get_ionization_indices(mols_sorted, mols_sorted[0])
        mols = []
        for nr_of_states, idx in enumerate(reaction_center_atom_idxs):
            logger.debug(Chem.MolToSmiles(mols_sorted[nr_of_states]))
            logger.debug(Chem.MolToSmiles(mols_sorted[nr_of_states + 1]))

            m = mol_to_paired_mol_data(
                mols_sorted[nr_of_states],
                mols_sorted[nr_of_states + 1],
                idx,
                selected_node_features,
                selected_edge_features,
            )
            loader = dataset_to_dataloader([m], 1)
            pka, pka_std = query_model.predict_pka_value(loader)
            pair = States(
                pka,
                pka_std,
                mols_sorted[nr_of_states],
                mols_sorted[nr_of_states + 1],
                idx,
                ph7_mol=mol_at_ph_7,
            )
            logger.debug("%s %s %s", pka, Chem.MolToSmiles(mols_sorted[nr_of_states]), Chem.MolToSmiles(mols_sorted[nr_of_states + 1]))
            mols.append(pair)
        logger.debug(mols)

    else:
        logger.info("Using dimorphite-dl to identify protonation sites.")
        mol_at_ph_7_list = _call_dimorphite_dl(mol, min_ph=7.0, max_ph=7.0, pka_precision=0)
        assert len(mol_at_ph_7_list) == 1
        mol_at_ph_7 = mol_at_ph_7_list[0]
        all_mols = _call_dimorphite_dl(mol, min_ph=0.5, max_ph=13.5)

        reaction_center_atom_idxs = sorted(list(set(_get_ionization_indices(all_mols, mol_at_ph_7))))
        mols = [mol_at_ph_7]

        acids = []
        mol_at_state = deepcopy(mol_at_ph_7)
        logger.info("Proposed mol at pH 7.4: %s", Chem.MolToSmiles(mol_at_state))

        used_reaction_center_atom_idxs = deepcopy(reaction_center_atom_idxs)
        logger.debug("Start with acids ...")
        for _ in reaction_center_atom_idxs:
            states_per_iteration = []
            for i in used_reaction_center_atom_idxs:
                try:
                    conj = create_conjugate(mol_at_state, i, pka=0.0, known_pka_values=False)
                except Exception:
                    continue

                logger.debug("%s", Chem.MolToSmiles(conj))
                sorted_mols = _sort_conj([conj, mol_at_state])
                m = mol_to_paired_mol_data(
                    sorted_mols[0],
                    sorted_mols[1],
                    i,
                    selected_node_features,
                    selected_edge_features,
                )
                loader = dataset_to_dataloader([m], 1)
                pka, pka_std = query_model.predict_pka_value(loader)
                pair = States(
                    pka,
                    pka_std,
                    sorted_mols[0],
                    sorted_mols[1],
                    reaction_center_idx=i,
                    ph7_mol=mol_at_ph_7,
                )

                if pka < 0.5:
                    logger.debug("Too low pKa value!")
                    continue

                logger.debug("acid: %s %s %s %s", pka, Chem.MolToSmiles(conj), i, Chem.MolToSmiles(mol_at_state))

                if acids:
                    if pka < acids[-1].pka:
                        states_per_iteration.append(pair)
                else:
                    states_per_iteration.append(pair)

            if not states_per_iteration:
                break

            acids.append(max(states_per_iteration, key=attrgetter("pka")))
            used_reaction_center_atom_idxs.remove(acids[-1].reaction_center_idx)
            mol_at_state = deepcopy(acids[-1].protonated_mol)

        logger.debug(acids)

        bases = []
        mol_at_state = deepcopy(mol_at_ph_7)
        logger.debug("Start with bases ...")
        used_reaction_center_atom_idxs = deepcopy(reaction_center_atom_idxs)
        logger.debug(reaction_center_atom_idxs)
        for _ in reaction_center_atom_idxs:
            states_per_iteration = []
            for i in used_reaction_center_atom_idxs:
                try:
                    conj = create_conjugate(mol_at_state, i, pka=13.5, known_pka_values=False)
                except Exception:
                    continue
                sorted_mols = _sort_conj([conj, mol_at_state])
                m = mol_to_paired_mol_data(
                    sorted_mols[0],
                    sorted_mols[1],
                    i,
                    selected_node_features,
                    selected_edge_features,
                )
                loader = dataset_to_dataloader([m], 1)
                pka, pka_std = query_model.predict_pka_value(loader)
                pair = States(
                    pka,
                    pka_std,
                    sorted_mols[0],
                    sorted_mols[1],
                    reaction_center_idx=i,
                    ph7_mol=mol_at_ph_7,
                )

                if pka > 13.5:
                    logger.debug("Too high pKa value!")
                    continue

                logger.debug("base: %s %s %s %s", pka, Chem.MolToSmiles(conj), i, Chem.MolToSmiles(mol_at_state))

                if bases:
                    if pka > bases[-1].pka:
                        states_per_iteration.append(pair)
                else:
                    states_per_iteration.append(pair)

            if not states_per_iteration:
                break

            bases.append(min(states_per_iteration, key=attrgetter("pka")))
            mol_at_state = deepcopy(bases[-1].deprotonated_mol)
            used_reaction_center_atom_idxs.remove(bases[-1].reaction_center_idx)

        logger.debug(bases)
        acids.reverse()
        mols = bases + acids
        mols = _check_for_duplicates(mols)

    if len(mols) == 0:
        logger.warning("Could not identify any ionizable group.")

    return mols


def calculate_pka_batch(
    smiles_list: List[str],
    query_model: Optional[QueryModel] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> List[Optional[List[States]]]:
    """Predict pKa values for a list of SMILES strings.

    Loads the model once and iterates, with per-molecule error handling so
    that a single failure does not abort the whole batch.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings.
    query_model : QueryModel, optional
        Pre-loaded model. Created once if not provided.
    device : torch.device or str, optional
        Compute device override.

    Returns
    -------
    list[list[States] | None]
        One entry per input SMILES. None if the molecule failed to process.
    """
    if query_model is None:
        query_model = QueryModel(device=device)

    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning("Invalid SMILES, skipping: %s", smi)
            results.append(None)
            continue
        try:
            states = calculate_microstate_pka_values(mol, query_model=query_model)
            results.append(states)
        except Exception as exc:
            logger.warning("Failed to process '%s': %s", smi, exc)
            results.append(None)
    return results


def draw_pka_map(protonation_states: list, size=(450, 450)):
    """Draw molecule at pH 7.0 with pKa values annotated at ionizable sites.

    Parameters
    ----------
    protonation_states : list[States]
        Output of calculate_microstate_pka_values.
    size : tuple[int, int]
        Image size in pixels.

    Returns
    -------
    PIL.Image
    """
    mol_at_ph_7 = deepcopy(protonation_states[0].ph7_mol)
    for protonation_state in range(len(protonation_states)):
        state = protonation_states[protonation_state]
        atom = mol_at_ph_7.GetAtomWithIdx(state.reaction_center_idx)
        try:
            atom.SetProp("atomNote", f'{atom.GetProp("atomNote")},   {state.pka:.2f}')
        except Exception:
            atom.SetProp("atomNote", f"{state.pka:.2f}")
    return Draw.MolToImage(mol_at_ph_7, size=size)


def draw_pka_reactions(
    protonation_states: list,
    height: int = 250,
    write_png_to_file: str = "",
):
    """Draw protonated/deprotonated pairs as a grid with equilibrium arrows and pKa values.

    Parameters
    ----------
    protonation_states : list[States]
        Output of calculate_microstate_pka_values.
    height : int
        Height per row in pixels.
    write_png_to_file : str
        If provided, write a PNG image to this file path.

    Returns
    -------
    IPython.display.SVG
        SVG display object (suitable for Jupyter notebooks).
    """
    from IPython.display import SVG

    draw_pairs, pair_atoms, legend = [], [], []
    for i in range(len(protonation_states)):
        state = protonation_states[i]
        draw_pairs.extend([state.protonated_mol, state.deprotonated_mol])
        pair_atoms.extend([[state.reaction_center_idx], [state.reaction_center_idx]])
        f = f"pka_{i} = {state.pka:.2f} (stddev: {state.pka_stddev:.2f})"
        legend.append(f)

    s = Draw.MolsToGridImage(
        draw_pairs,
        molsPerRow=2,
        subImgSize=(height * 2, height),
        highlightAtomLists=pair_atoms,
        useSVG=True,
    )
    if hasattr(s, "data"):
        s = s.data.replace("svg:", "")
    fig = sg.fromstring(s)
    for i, text in enumerate(legend):
        label = sg.TextElement(
            height * 2,
            (height * (i + 1)) - 10,
            text,
            size=14,
            font="sans-serif",
            anchor="middle",
        )
        fig.append(label)
        h = height * (i + 0.5)
        w = height * 2
        fig.append(
            sg.LineElement(
                [(w * 0.9, h - height * 0.02), (w * 1.1, h - height * 0.02)],
                width=2,
                color="black",
            )
        )
        fig.append(
            sg.LineElement(
                [(w * 1.1, h - height * 0.02), (w * 1.07, h - height * 0.04)],
                width=2,
                color="black",
            )
        )
        fig.append(
            sg.LineElement(
                [(w * 0.9, h + height * 0.02), (w * 1.1, h + height * 0.02)],
                width=2,
                color="black",
            )
        )
        fig.append(
            sg.LineElement(
                [(w * 0.9, h + height * 0.02), (w * 0.93, h + height * 0.04)],
                width=2,
                color="black",
            )
        )
    if write_png_to_file:
        cairosvg.svg2png(bytestring=fig.to_str(), write_to=f"{write_png_to_file}", dpi=300)
    return SVG(fig.to_str())
