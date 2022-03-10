# imports
import logging
from copy import deepcopy
from dataclasses import dataclass
from operator import attrgetter
from os import path

import cairosvg
import numpy as np
import svgutils.transform as sg
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from torch_geometric.loader import DataLoader

from pkasolver.chem import create_conjugate
from pkasolver.constants import DEVICE, EDGE_FEATURES, NODE_FEATURES
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
    def __init__(self):

        self.models = []

        for i in range(25):
            model_name, model_class = "GINPair", GINPairV1
            model = model_class(
                num_node_features, num_edge_features, hidden_channels=96
            )
            base_path = path.dirname(__file__)
            if torch.cuda.is_available() == False:  # If only CPU is available
                checkpoint = torch.load(
                    f"{base_path}/trained_model_without_epik/best_model_{i}.pt",
                    map_location=torch.device("cpu"),
                )
            else:
                checkpoint = torch.load(
                    f"{base_path}/trained_model_without_epik/best_model_{i}.pt"
                )

            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            model.to(device=DEVICE)
            self.models.append(model)

    def predict_pka_value(self, loader: DataLoader) -> np.ndarray:
        """
        ----------
        loader
            data to be predicted
        Returns
        -------
        np.array
            list of predicted pKa values
        """

        results = []
        assert len(loader) == 1
        for data in loader:  # Iterate in batches over the training dataset.
            data.to(device=DEVICE)
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
    and returns the protonation center index

    """
    from rdkit.Chem import rdFMCS

    list_of_reaction_centers = []
    for idx, m2 in enumerate(mol_list):

        m1 = compare_to
        assert m1.GetNumAtoms() == m2.GetNumAtoms()

        # find MCS
        mcs = rdFMCS.FindMCS(
            [m1, m2],
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            timeout=120,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
        )

        # convert from SMARTS
        mcsp = Chem.MolFromSmarts(mcs.smartsString, False)
        s1 = m1.GetSubstructMatch(mcsp)
        s2 = m2.GetSubstructMatch(mcsp)

        for i, j in zip(s1, s2):
            if i != j:  # matching not sucessfull
                break
            if (
                m1.GetAtomWithIdx(i).GetFormalCharge()
                != m2.GetAtomWithIdx(j).GetFormalCharge()
            ):
                if i != j:  # matching not sucessfull
                    logger.warning("Trouble ahead ... different atom indices detected.")
                list_of_reaction_centers.append(i)

    logger.debug(set(list_of_reaction_centers))
    return list_of_reaction_centers


def _parse_dimorphite_dl_output():
    import pickle

    mols = pickle.load(open("test.pkl", "rb"))
    return mols


def _call_dimorphite_dl(
    mol: Chem.Mol, min_ph: float, max_ph: float, pka_precision: float = 1.0
):
    """calls  dimorphite_dl with parameters"""
    import subprocess

    # get path to script
    path_to_script = path.dirname(__file__)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    # save properties
    props = mol.GetPropsAsDict()

    o = subprocess.run(
        [
            "python",
            f"{path_to_script}/scripts/call_dimorphite_dl.py",
            "--smiles",  # only most probable tautomer generated
            f"{smiles}",  # don't adjust the ionization state of the molecule
            "--min_ph",
            f"{min_ph}",
            "--max_ph",
            f"{max_ph}",
            "--pka_precision",
            f"{pka_precision}",
        ],
        stderr=subprocess.STDOUT,
    )
    o.check_returncode()
    # get list of smiles
    mols = _parse_dimorphite_dl_output()
    return mols


def _sort_conj(mols: list):
    """sort mols based on number of hydrogen"""

    assert len(mols) == 2
    nr_of_hydrogen = [
        np.sum([atom.GetTotalNumHs() for atom in mol.GetAtoms()]) for mol in mols
    ]
    if abs(nr_of_hydrogen[0] - nr_of_hydrogen[1]) != 1:
        raise RuntimeError(
            "Neighboring protonation states are only allowed to have a difference of a single hydrogen."
        )
    mols_sorted = [
        x for _, x in sorted(zip(nr_of_hydrogen, mols), reverse=True)
    ]  # https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
    return mols_sorted


def _check_for_duplicates(states: list):
    """check whether two states have the same pKa value and remove one of them"""
    all_r = dict()
    logger.debug(states)
    for state in states:
        m1, m2 = _sort_conj([state.protonated_mol, state.deprotonated_mol])
        all_r[hash((Chem.MolToSmiles(m1), Chem.MolToSmiles(m2)))] = state
    # logger.debug([all_r[k] for k in sorted(all_r, key=all_r.get)])
    return sorted([all_r[k] for k in all_r], key=attrgetter("pka"))


def calculate_microstate_pka_values(
    mol: Chem.rdchem.Mol, only_dimorphite: bool = False, query_model=None
):
    """Enumerate protonation states using a rdkit mol as input"""

    if query_model == None:
        query_model = QueryModel()

    if only_dimorphite:
        print(
            "BEWARE! This is experimental and might generate wrong protonation states."
        )
        logger.debug("Using dimorphite-dl to enumerate protonation states.")
        mol_at_ph_7 = _call_dimorphite_dl(mol, min_ph=7.0, max_ph=7.0, pka_precision=0)
        all_mols = _call_dimorphite_dl(mol, min_ph=0.5, max_ph=13.5)
        # sort mols
        atom_charges = [
            np.sum([atom.GetTotalNumHs() for atom in mol.GetAtoms()])
            for mol in all_mols
        ]

        mols_sorted = [
            x for _, x in sorted(zip(atom_charges, all_mols), reverse=True)
        ]  # https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list

        reaction_center_atom_idxs = _get_ionization_indices(mols_sorted, mols_sorted[0])
        # return only mol pairs
        mols = []
        for nr_of_states, idx in enumerate(reaction_center_atom_idxs):
            logger.debug(Chem.MolToSmiles(mols_sorted[nr_of_states]))
            logger.debug(Chem.MolToSmiles(mols_sorted[nr_of_states + 1]))

            # generated paired data structure
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
            logger.debug(
                pka,
                Chem.MolToSmiles(mols_sorted[nr_of_states]),
                Chem.MolToSmiles(mols_sorted[nr_of_states + 1]),
            )

            mols.append(pair)
        logger.debug(mols)

    else:
        logger.info("Using dimorphite-dl to identify protonation sites.")
        mol_at_ph_7 = _call_dimorphite_dl(mol, min_ph=7.0, max_ph=7.0, pka_precision=0)
        assert len(mol_at_ph_7) == 1
        mol_at_ph_7 = mol_at_ph_7[0]
        all_mols = _call_dimorphite_dl(mol, min_ph=0.5, max_ph=13.5)

        # identify protonation sites
        reaction_center_atom_idxs = sorted(
            list(set(_get_ionization_indices(all_mols, mol_at_ph_7)))
        )
        mols = [mol_at_ph_7]

        acids = []
        mol_at_state = deepcopy(mol_at_ph_7)
        print(f"Proposed mol at pH 7.4: {Chem.MolToSmiles(mol_at_state)}")

        used_reaction_center_atom_idxs = deepcopy(reaction_center_atom_idxs)
        logger.debug("Start with acids ...")
        # for each possible protonation state
        for _ in reaction_center_atom_idxs:
            states_per_iteration = []
            # for each possible reaction center
            for i in used_reaction_center_atom_idxs:
                try:
                    conj = create_conjugate(
                        mol_at_state, i, pka=0.0, known_pka_values=False,
                    )
                except:
                    continue

                logger.debug(f"{Chem.MolToSmiles(conj)}")

                # sort mols (protonated/deprotonated)
                sorted_mols = _sort_conj([conj, mol_at_state])

                m = mol_to_paired_mol_data(
                    sorted_mols[0],
                    sorted_mols[1],
                    i,
                    selected_node_features,
                    selected_edge_features,
                )
                # calc pka value
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

                # test if pka is inside pH range
                if pka < 0.5:
                    logger.debug("Too low pKa value!")
                    # skip rest
                    continue

                logger.debug(
                    "acid: ",
                    pka,
                    Chem.MolToSmiles(conj),
                    i,
                    Chem.MolToSmiles(mol_at_state),
                )

                # if this is NOT the first state found
                if acids:
                    # check if previous pka value is lower and if yes, add it
                    if pka < acids[-1].pka:
                        states_per_iteration.append(pair)
                else:
                    # if this is the first state found
                    states_per_iteration.append(pair)

            if not states_per_iteration:
                # no protonation state left
                break

            # get the protonation state with the highest pka
            acids.append(max(states_per_iteration, key=attrgetter("pka")))
            used_reaction_center_atom_idxs.remove(
                acids[-1].reaction_center_idx
            )  # avoid double protonation
            mol_at_state = deepcopy(acids[-1].protonated_mol)

        logger.debug(acids)

        #######################################################
        # continue with bases
        #######################################################

        bases = []
        mol_at_state = deepcopy(mol_at_ph_7)
        logger.debug("Start with bases ...")
        used_reaction_center_atom_idxs = deepcopy(reaction_center_atom_idxs)
        logger.debug(reaction_center_atom_idxs)
        # for each possible protonation state
        for _ in reaction_center_atom_idxs:
            states_per_iteration = []
            # for each possible reaction center
            for i in used_reaction_center_atom_idxs:
                try:
                    conj = create_conjugate(
                        mol_at_state, i, pka=13.5, known_pka_values=False
                    )
                except:
                    continue
                sorted_mols = _sort_conj([conj, mol_at_state])
                m = mol_to_paired_mol_data(
                    sorted_mols[0],
                    sorted_mols[1],
                    i,
                    selected_node_features,
                    selected_edge_features,
                )
                # calc pka values
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

                # check if pka is within pH range
                if pka > 13.5:
                    logger.debug("Too high pKa value!")
                    continue

                logger.debug(
                    "base",
                    pka,
                    Chem.MolToSmiles(conj),
                    i,
                    Chem.MolToSmiles(mol_at_state),
                )
                # if bases already present
                if bases:
                    # check if previous pka is higher
                    if pka > bases[-1].pka:
                        states_per_iteration.append(pair)
                else:
                    states_per_iteration.append(pair)

            if not states_per_iteration:
                # no protonation state left
                break
            # take state with lowest pka value
            bases.append(min(states_per_iteration, key=attrgetter("pka")))
            mol_at_state = deepcopy(bases[-1].deprotonated_mol)
            used_reaction_center_atom_idxs.remove(
                bases[-1].reaction_center_idx
            )  # avoid double deprotonation

        logger.debug(bases)
        acids.reverse()
        mols = bases + acids
        # remove possible duplications
        mols = _check_for_duplicates(mols)

    if len(mols) == 0:
        print("#########################")
        print("Could not identify any ionizable group. Aborting.")
        print("#########################")

    return mols


def draw_pka_map(protonation_states: list, size=(450, 450)):
    """draw mol at pH=7.0 and indicate protonation sites with respectiv pKa values"""
    mol_at_ph_7 = deepcopy(protonation_states[0].ph7_mol)
    for protonation_state in range(len(protonation_states)):

        state = protonation_states[protonation_state]
        atom = mol_at_ph_7.GetAtomWithIdx(state.reaction_center_idx)
        try:
            atom.SetProp("atomNote", f'{atom.GetProp("atomNote")},   {state.pka:.2f}')
        except:
            atom.SetProp("atomNote", f"{state.pka:.2f}")
    return Draw.MolToImage(mol_at_ph_7, size=size)


def draw_pka_reactions(
    protonation_states: list, height=250, write_png_to_file: str = ""
):
    """
    Draws protonation states.
    file can be saved as png using `write_png_to_file` parameter.
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
    if hasattr(
        s, "data"
    ):  # Draw.MolsToGridImage returns different output depending on whether it is called in a notebook or a script
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
    # if png file path is passed write png file
    if write_png_to_file:
        cairosvg.svg2png(
            bytestring=fig.to_str(), write_to=f"{write_png_to_file}", dpi=300
        )
    return SVG(fig.to_str())
