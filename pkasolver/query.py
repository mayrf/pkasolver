# imports
import logging
from os import path

import numpy as np
from IPython.display import display
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole

# IPythonConsole.drawOptions.addAtomIndices = True
IPythonConsole.molSize = 400, 400
from copy import deepcopy

import torch
from rdkit import RDLogger
from rdkit.Chem import Draw

from pkasolver.chem import create_conjugate
from pkasolver.constants import DEVICE, EDGE_FEATURES, NODE_FEATURES
from pkasolver.data import (
    calculate_nr_of_features,
    make_features_dicts,
    mol_to_paired_mol_data,
)
from pkasolver.ml import dataset_to_dataloader, predict_pka_value
from pkasolver.ml_architecture import GINPairV1

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
    def __init__(self, path_to_parameters: str = ""):

        model_name, model_class = "GINPair", GINPairV1
        model = model_class(num_node_features, num_edge_features, hidden_channels=96)

        if path_to_parameters:
            checkpoint = torch.load(path_to_parameters)
        else:
            base_path = path.dirname(__file__)
            checkpoint = torch.load(f"{base_path}/trained_model/fine_tuned_model.pt")

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(device=DEVICE)
        self.model = model


def get_ionization_indices(mol_list: list, strict: bool = True) -> list:
    """Takes a list of mol objects of different protonation states,
    and returns the protonation center index

    """
    from rdkit.Chem import rdFMCS

    list_of_reaction_centers = []

    for idx, m1 in enumerate(mol_list):
        if idx == len(mol_list) - 1:
            # skip last
            continue
        m2 = mol_list[idx + 1]
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
        print("searching ...")
        s1 = m1.GetSubstructMatch(mcsp)
        s2 = m2.GetSubstructMatch(mcsp)
        for i, j in zip(s1, s2):
            if (
                m1.GetAtomWithIdx(i).GetFormalCharge()
                != m2.GetAtomWithIdx(j).GetFormalCharge()
            ):
                if i != j:
                    logger.warning("Trouble ahead ... different atom indices detected.")
                    continue
                list_of_reaction_centers.append(i)

    if strict:
        print(list_of_reaction_centers)
        assert len(list_of_reaction_centers) == len(mol_list) - 1

    return list_of_reaction_centers


def get_possible_reactions(mol, matches):
    acid_pairs = []
    base_pairs = []
    for match in matches:
        print(match)
        mol.__sssAtoms = [match]  # not sure if needed
        # create conjugate
        new_mol = deepcopy(mol)
        atom = new_mol.GetAtomWithIdx(match)
        element = atom.GetAtomicNum()
        charge = atom.GetFormalCharge()
        Ex_Hs = atom.GetNumExplicitHs()
        Tot_Hs = atom.GetTotalNumHs()
        if (element == 7 and charge <= 0) or charge < 0:
            # increase H
            try:
                atom.SetFormalCharge(charge + 1)
                if Tot_Hs == 0 or Ex_Hs > 0:
                    atom.SetNumExplicitHs(Ex_Hs + 1)
                atom.UpdatePropertyCache()
                acid_pairs.append((new_mol, mol, match))
            except:
                pass

        # reset changes in case atom can also be deprotonated
        new_mol = deepcopy(mol)
        atom = new_mol.GetAtomWithIdx(match)
        element = atom.GetAtomicNum()
        charge = atom.GetFormalCharge()
        Ex_Hs = atom.GetNumExplicitHs()
        Tot_Hs = atom.GetTotalNumHs()

        if Tot_Hs > 0 and charge >= 0:
            # reduce H
            atom.SetFormalCharge(charge - 1)
            if Ex_Hs > 0:
                atom.SetNumExplicitHs(Ex_Hs - 1)
            atom.UpdatePropertyCache()
            base_pairs.append((mol, new_mol, match))

    return acid_pairs, base_pairs


def match_pka(pair_tuples, model):
    pair_data = []
    for (prot, deprot, atom_idx) in pair_tuples:
        m = mol_to_paired_mol_data(
            prot, deprot, atom_idx, selected_node_features, selected_edge_features,
        )
        pair_data.append(m)
    loader = dataset_to_dataloader(pair_data, 1, shuffle=False)
    return np.round(predict_pka_value(model, loader), 3)


def acid_sequence(acid_pairs, mols, pkas, atoms):
    # determine pka for protonatable groups
    query_model = QueryModel()

    if len(acid_pairs) > 0:
        acid_pkas = list(match_pka(acid_pairs, query_model.model))
        pka = max(acid_pkas)  # determining closest protonation pka
        if pka < 0.5:  # do not include pka if lower than 0.5
            return mols, pkas, atoms

        pkas.insert(0, pka)  # prepending pka to global pka list
        mols.insert(
            0, acid_pairs[acid_pkas.index(pka)][0]
        )  # prepending protonated molcule to global mol list
        atoms.insert(
            0, acid_pairs[acid_pkas.index(pka)][2]
        )  # prepending protonated molcule to global mol list
    return mols, pkas, atoms


def base_sequence(base_pairs, mols, pkas, atoms):
    query_model = QueryModel()
    # determine pka for deprotonatable groups
    if len(base_pairs) > 0:
        base_pkas = list(match_pka(base_pairs, query_model.model))
        pka = min(base_pkas)  # determining closest deprotonation pka
        if pka > 13.5:  # do not include if pka higher than 13.5
            return mols, pkas, atoms
        pkas.append(pka)  # appending pka to global pka list
        mols.append(
            base_pairs[base_pkas.index(pka)][1]
        )  # appending protonated molcule to global mol list
        atoms.append(
            base_pairs[base_pkas.index(pka)][2]
        )  # appending protonated molcule to global mol list
    return mols, pkas, atoms


def _parse_dimorphite_dl_output():

    print("parsing ...")
    mols = []
    with open("output.smi", "r") as f:
        for line in f:
            if line:
                print(line)
                mols.append(Chem.MolFromSmiles(line))
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
            f"{path_to_script}/dimorphite_dl/dimorphite_dl.py",  # call dimorphite_dl
            "--smiles",  # only most probable tautomer generated
            f"{smiles}",  # don't adjust the ionization state of the molecule
            "--min_ph",
            f"{min_ph}",
            "--max_ph",
            f"{max_ph}",
            "--pka_precision",
            f"{pka_precision}",
            "--output_file",
            "output.smi",
            "--label_states",
        ],
        stderr=subprocess.STDOUT,
    )
    o.check_returncode()

    # get list of smiles
    mols = _parse_dimorphite_dl_output()
    print(mols)

    # insert properties in newly generated mols
    for m in mols:
        for prop, val in props.items():
            if type(val) is int:
                m.SetIntProp(prop, val)
            elif type(val) is float:
                m.SetDoubleProp(prop, val)
            elif type(val) is bool:
                m.SetBoolProp(prop, val)
            else:
                m.SetProp(prop, str(val))
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


def _check_for_dublicates(states: list):
    """check whether two states have the same pKa value and remove one of them"""
    all_r = dict()
    for state in states:
        m1, m2 = _sort_conj([state[1][0], state[1][1]])
        all_r[hash((Chem.MolToSmiles(m1), Chem.MolToSmiles(m2)))] = state
    logger.debug([all_r[k] for k in sorted(all_r, key=all_r.get)])
    return [all_r[k] for k in sorted(all_r, key=all_r.get)]


def calculate_microstate_pka_values(mol: Chem.rdchem.Mol, only_dimorphite: bool = True):
    """Enumerate protonation states using a rdkit mol as input"""
    from operator import itemgetter

    if only_dimorphite:
        query_model = QueryModel()
        print("Using dimorphite-dl to enumerate protonation states.")
        all_mols = _call_dimorphite_dl(mol, min_ph=0.5, max_ph=13.5)
        # sort mols
        atom_charges = [
            np.sum([atom.GetTotalNumHs() for atom in mol.GetAtoms()])
            for mol in all_mols
        ]

        mols_sorted = [
            x for _, x in sorted(zip(atom_charges, all_mols), reverse=True)
        ]  # https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list

        reaction_center_atom_idxs = get_ionization_indices(mols_sorted)
        # return only mol pairs
        mols = []
        for nr_of_states, idx in enumerate(reaction_center_atom_idxs):
            print(Chem.MolToSmiles(mols_sorted[nr_of_states]))
            print(Chem.MolToSmiles(mols_sorted[nr_of_states + 1]))

            m = mol_to_paired_mol_data(
                mols_sorted[nr_of_states],
                mols_sorted[nr_of_states + 1],
                idx,
                selected_node_features,
                selected_edge_features,
            )
            loader = dataset_to_dataloader([m], 1)
            pka = predict_pka_value(query_model.model, loader)
            pair = (
                pka,
                (mols_sorted[nr_of_states], mols_sorted[nr_of_states + 1]),
                idx,
            )
            logger.debug(
                pka,
                Chem.MolToSmiles(mols_sorted[nr_of_states]),
                Chem.MolToSmiles(mols_sorted[nr_of_states + 1]),
            )

            mols.append(pair)

        print(mols)

    else:
        print("Using dimorphite-dl to identify protonation sites.")
        mol_at_ph_7 = _call_dimorphite_dl(mol, min_ph=7.0, max_ph=7.0, pka_precision=0)[
            0
        ]
        all_mols = _call_dimorphite_dl(mol, min_ph=0.5, max_ph=13.5)

        reaction_center_atom_idxs = sorted(
            list(set(get_ionization_indices(all_mols, strict=False)))
        )
        mols = [mol_at_ph_7]
        query_model = QueryModel()

        acids = []
        mol_at_state = deepcopy(mol_at_ph_7)
        print(f"Mol at pH 7.4: {Chem.MolToSmiles(mol_at_state)}")

        used_reaction_center_atom_idxs = deepcopy(reaction_center_atom_idxs)
        logger.debug("Start with acids ...")
        for _ in reaction_center_atom_idxs:
            logger.debug("Acid groups ...")
            states_per_iteration = []
            for i in used_reaction_center_atom_idxs:
                try:
                    conj = create_conjugate(
                        mol_at_state,
                        i,
                        pka=0.0,
                        known_pka_values=False,
                    )
                except:
                    continue
                logger.debug(f"{Chem.MolToSmiles(conj)}")

                sorted_mols = _sort_conj([conj, mol_at_state])

                m = mol_to_paired_mol_data(
                    sorted_mols[0],
                    sorted_mols[1],
                    i,
                    selected_node_features,
                    selected_edge_features,
                )
                loader = dataset_to_dataloader([m], 1)
                pka = predict_pka_value(query_model.model, loader)[0]
                if pka < 0.5:
                    logger.debug("Too low pKa value!")
                    continue
                print(
                    "acid: ",
                    pka,
                    Chem.MolToSmiles(conj),
                    i,
                    Chem.MolToSmiles(mol_at_state),
                )
                if acids:
                    if pka < acids[-1][0]:
                        states_per_iteration.append((pka, (conj, mol_at_state), i))
                else:
                    states_per_iteration.append((pka, (conj, mol_at_state), i))

            if not states_per_iteration:
                # no protonation state left
                break

            acids.append(max(states_per_iteration, key=itemgetter(0)))
            used_reaction_center_atom_idxs.remove(
                acids[-1][2]
            )  # avoid double protonation
            mol_at_state = deepcopy(acids[-1][1][0])

        logger.debug(acids)

        bases = []
        mol_at_state = deepcopy(mol_at_ph_7)
        logger.debug("Start with bases ...")
        used_reaction_center_atom_idxs = deepcopy(reaction_center_atom_idxs)
        for _ in reaction_center_atom_idxs:
            states_per_iteration = []
            for i in reaction_center_atom_idxs:
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
                loader = dataset_to_dataloader([m], 1)
                pka = predict_pka_value(query_model.model, loader)[0]
                if pka > 13.5:
                    logger.debug("Too high pKa value!")
                    continue

                print(
                    "base",
                    pka,
                    Chem.MolToSmiles(conj),
                    i,
                    Chem.MolToSmiles(mol_at_state),
                )
                if bases:
                    if pka > bases[-1][0]:
                        states_per_iteration.append((pka, (mol_at_state, conj), i))
                else:
                    states_per_iteration.append((pka, (mol_at_state, conj), i))

            if not states_per_iteration:
                # no protonation state left
                break
            bases.append(min(states_per_iteration, key=itemgetter(0)))
            mol_at_state = deepcopy(bases[-1][1][1])
            used_reaction_center_atom_idxs.remove(
                bases[-1][2]
            )  # avoid double deprotonation

        logger.debug(bases)
        acids.reverse()
        mols = bases + acids

        mols = _check_for_dublicates(mols)

    if len(mols) == 0:
        print('#########################')
        print ('Could not identify any ionizable group. Aborting.')
        print('#########################')

    return mols


def draw_pka_map(mols, pkas, atoms, size=(450, 450)):
    mol = mols[0][0]
    for atom, pka in zip(atoms, pkas):
        mol.GetAtomWithIdx(atom).SetProp("atomNote", f"{pka:.2f}")
    return Draw.MolToImage(mol, size=size)


def draw_pka_reactions(mols, pkas, atoms):
    draw_pairs = []
    pair_atoms = []
    pair_pkas = []
    for i in range(len(mols)):
        draw_pairs.extend([mols[i][0], mols[i][1]])
        pair_atoms.extend([[atoms[i]], [atoms[i]]])
        pair_pkas.extend([pkas[i], pkas[i]])
    return Draw.MolsToGridImage(
        draw_pairs,
        molsPerRow=2,
        subImgSize=(250, 250),
        highlightAtomLists=pair_atoms,
        legends=[f"pka = {pair_pkas[i]:.2f}" for i in range(12)],
    )


def draw_sdf_mols(input_path, range_list=[]):
    print(f"opening .sdf file at {input_path} and computing pkas...")
    with open(input_path, "rb") as fh:
        count = 0
        for i, mol in enumerate(Chem.ForwardSDMolSupplier(fh, removeHs=True)):
            if range_list and i not in range_list:
                continue
            props = mol.GetPropsAsDict()
            for prop in props.keys():
                mol.ClearProp(prop)
            mols, pkas, atoms = calculate_microstate_pka_values(mol)
            display(draw_pka_map(mols, pkas, atoms))
            print(f"Name: {mol.GetProp('_Name')}")
