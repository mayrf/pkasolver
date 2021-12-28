# imports
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import AllChem
from IPython.display import display
# IPythonConsole.drawOptions.addAtomIndices = True
IPythonConsole.molSize = 400,400
from pkasolver.ml_architecture import GINPairV1, GINPairV2
from pkasolver.ml import (
    dataset_to_dataloader,
    predict
)
from pkasolver.data import calculate_nr_of_features, mol_to_paired_mol_data, make_features_dicts
import torch
import torch_geometric
from copy import deepcopy
from rdkit.Chem import Draw
from pkasolver.constants import EDGE_FEATURES, NODE_FEATURES, DEVICE
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


    
# paths and constants
smarts_file = "smarts_pattern_impl.tsv"
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

model_path = "/data/shared/projects/pkasolver-data-clean/trained_models_v1/training_with_GINPairV1_v1_hp/reg_everything_best_model.pt"

class QueryModel:
    
    def __init__(self, path):
        self.path = path
        self.model_init()

    def model_init(self):
        self.model_name, self.model_class = "GINPairV1", GINPairV1
        self.model = self.model_class(
                    num_node_features, num_edge_features, hidden_channels=96
                )
        self.checkpoint = torch.load(self.path)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()
        self.model.to(device = DEVICE)
    

    def set_path(self, new_path):
        self.path = new_path
        self.model_init()

query_model = QueryModel(model_path)

    # helper functions

def set_model_path(new_path, query_model=query_model):
    query_model.set_path(new_path)

def split_acid_base_pattern(smarts_file): # from molGpka
    df_smarts = pd.read_csv(smarts_file, sep="\t")
    df_smarts_acid = df_smarts[df_smarts.Acid_or_base == "A"]
    df_smarts_base = df_smarts[df_smarts.Acid_or_base == "B"]
    return df_smarts_acid, df_smarts_base
    
def unique_acid_match(matches):     # from molGpka
    single_matches = list(set([m[0] for m in matches if len(m)==1]))
    double_matches = [m for m in matches if len(m)==2]
    single_matches = [[j] for j in single_matches]
    double_matches.extend(single_matches)
    return double_matches

def match_acid(df_smarts_acid, mol):     # from molGpka
    matches = []
    for idx, name, smarts, index, acid_base in df_smarts_acid.itertuples():
        pattern = Chem.MolFromSmarts(smarts)
        match = mol.GetSubstructMatches(pattern)
        if len(match) == 0:
            continue
        if len(index) > 2:
            index = index.split(",")
            index = [int(i) for i in index]
            for m in match:
                matches.append([m[index[0]], m[index[1]]])
        else:
            index = int(index)
            for m in match:
                matches.append([m[index]])
    matches = unique_acid_match(matches)
    matches_modify = []
    for i in matches:
        for j in i:
            matches_modify.append(j)
    return matches_modify

def match_base(df_smarts_base, mol):     # from molGpka
    matches = []
    for idx, name, smarts, index, acid_base in df_smarts_base.itertuples():
        pattern = Chem.MolFromSmarts(smarts)
        match = mol.GetSubstructMatches(pattern)
        if len(match) == 0:
            continue
        if len(index) > 2:
            index = index.split(",")
            index = [int(i) for i in index]
            for m in match:
                matches.append([m[index[0]], m[index[1]]])
        else:
            index = int(index)
            for m in match:
                matches.append([m[index]])
    matches = unique_acid_match(matches)
    matches_modify = []
    for i in matches:
        for j in i:
            matches_modify.append(j)
    return matches_modify

def get_ionization_aid(mol, acid_or_base=None):     # from molGpka
    df_smarts_acid, df_smarts_base = split_acid_base_pattern(smarts_file)

    if mol == None:
        raise RuntimeError("read mol error: {}".format(mol_file))
    acid_matches = match_acid(df_smarts_acid, mol)
    base_matches = match_base(df_smarts_base, mol)
    if acid_or_base == None:
        return acid_matches, base_matches
    elif acid_or_base == "acid":
        return acid_matches
    else:
        return base_matches


df_smarts_acid, df_smarts_base = split_acid_base_pattern(smarts_file)



def get_possible_reactions(mol):

    matches = get_ionization_aid(mol)
    matches = sum(matches, []) # flatten matches list

    acid_pairs = []
    base_pairs = []
    for match in matches:
        is_prot = True
        mol.__sssAtoms = [match]
        new_mol = deepcopy(mol)
        # create conjugate
        atom = new_mol.GetAtomWithIdx(match)
        charge = atom.GetFormalCharge()
        Ex_Hs = atom.GetNumExplicitHs()
        Tot_Hs = atom.GetTotalNumHs()
        if Tot_Hs > 0 and charge >= 0:
            # reduce H
            atom.SetFormalCharge(charge - 1)
            if Ex_Hs > 0:
                atom.SetNumExplicitHs(Ex_Hs - 1)
        elif Tot_Hs == 0 or charge < 0:
            # increase H
            atom.SetFormalCharge(charge + 1)
            if Tot_Hs == 0 or Ex_Hs > 0:
                atom.SetNumExplicitHs(Ex_Hs + 1)
            is_prot = False
        
        atom.UpdatePropertyCache()
        
        # add tuple of conjugates
        if is_prot:
            base_pairs.append((mol, new_mol, match))
        else:
            acid_pairs.append((new_mol, mol, match))
    return acid_pairs, base_pairs

def match_pka(pair_tuples, model):
    pair_data = []
    for (prot, deprot, atom_idx) in pair_tuples:
        m = mol_to_paired_mol_data(
                prot,
                deprot,
                atom_idx,
                selected_node_features,
                selected_edge_features,
            )
        pair_data.append(m)
    loader = dataset_to_dataloader(pair_data, 64, shuffle=False)
    return predict(model, loader)

def acid_sequence(acid_pairs, mols, pkas, atoms):
    # determine pka for protonatable groups
    if len(acid_pairs) > 0:
        acid_pkas = list(match_pka(acid_pairs, query_model.model))
        pka = max(acid_pkas)    # determining closest protonation pka
        pkas.insert(0, pka)     # prepending pka to global pka list
        mols.insert(0, acid_pairs[acid_pkas.index(pka)][0]) # prepending protonated molcule to global mol list 
        atoms.insert(0, acid_pairs[acid_pkas.index(pka)][2]) # prepending protonated molcule to global mol list 
    return mols, pkas, atoms

def base_sequence(base_pairs, mols, pkas, atoms):
    # determine pka for deprotonatable groups
    if len(base_pairs) > 0:
        base_pkas = list(match_pka(base_pairs, query_model.model))
        pka = min(base_pkas)    # determining closest deprotonation pka   
        pkas.append(pka)        # appending pka to global pka list    
        mols.append(base_pairs[base_pkas.index(pka)][1]) # appending protonated molcule to global mol list 
        atoms.append(base_pairs[base_pkas.index(pka)][2]) # appending protonated molcule to global mol list 
    return mols, pkas, atoms

def mol_query(mol: Chem.rdchem.Mol):

    mols = [mol]
    pkas = []
    atoms = []

    while True:
        inital_length = len(pkas) 
        acid_pairs, base_pairs = get_possible_reactions(mols[0])
        mols, pkas, atoms = acid_sequence(acid_pairs, mols, pkas, atoms)
        if inital_length >= len(pkas):
            break
    while True:
        inital_length = len(pkas) 
        acid_pairs, base_pairs = get_possible_reactions(mols[-1])
        mols, pkas, atoms = base_sequence(base_pairs, mols, pkas, atoms)
        if inital_length >= len(pkas):
            break
    
    mol_tuples = []
    for i in range(len(mols)-1):
        mol_tuples.append((mols[i],mols[i+1]))
    mols = mol_tuples

    return mols, pkas, atoms

def smiles_query(smi, output_smiles = False):
    mols, pkas, atoms = mol_query(Chem.MolFromSmiles(smi))
    if output_smiles == True:
        smiles = []
        for mol in mols:
            smiles.append((Chem.MolToSmiles(mol[0]),Chem.MolToSmiles(mol[1])))
        mols = smiles
    return mols, pkas, atoms

def inchi_query(ini, output_inchi = False):
    # return mol_query(Chem.MolFromInchi(ini))
    mols, pkas, atoms = mol_query(Chem.MolFromInchi(ini))
    if output_inchi == True:
        inchi = []
        for mol in mols:
            inchi.append((Chem.MolToInchi(mol[0]),Chem.MolToInchi(mol[1])))
        mols = inchi 
    return mols, pkas, atoms

def sdf_query(input_path, output_path, merged_output=False):
    print(f"opening .sdf file at {input_path} and computing pkas...")
    with open(input_path, "rb") as fh:
        with open(output_path, "w") as sdf_zip:
            with Chem.SDWriter(sdf_zip) as writer:  
                count = 0
                for i, mol in enumerate(Chem.ForwardSDMolSupplier(fh, removeHs=True)):
                    # if i > 10: 
                    #     break
                    # clear porps
                    props = mol.GetPropsAsDict()
                    for prop in props.keys():
                        mol.ClearProp(prop)
                    mols, pkas, atoms = mol_query(mol)
                    if merged_output == True:
                        mol = mols[0][0]
                        mol.SetProp("ID", f"{mol.GetProp('_Name')}")
                        for ii, (pka, atom) in enumerate(zip(pkas, atoms)):
                            count += 1
                            mol.SetProp(f"pka_{ii}", f"{pka}")
                            mol.SetProp(f"atom_idx_{ii}", f"{atom}")
                            writer.write(mol)
                    else:   
                        for ii, (mol, pka, atom) in enumerate(zip(mols, pkas, atoms)):
                            count += 1
                            mol = mol[0]
                            mol.SetProp("ID", f"{mol.GetProp('_Name')}_{ii}")
                            mol.SetProp("pka", f"{pka}")
                            mol.SetProp("atom_idx", f"{atom}")
                            mol.SetProp("pka-number", f"{ii}")
                            # print(mol.GetPropsAsDict())
                            writer.write(mol)
                print(f"{count} pkas for {i} molecules predicted and saved at \n{output_path}")

def draw_pka_map(mols, pkas, atoms, size=(450,450)):
    mol = mols[0][0]
    for atom, pka in zip(atoms, pkas):
        mol.GetAtomWithIdx(atom).SetProp('atomNote', f"{pka:.2f}" )
    return Draw.MolToImage(mol, size=size)

def draw_pka_reactions(mols, pkas, atoms):
    draw_pairs = [] 
    pair_atoms = []
    pair_pkas = []
    for i in range(len(mols)):
        draw_pairs.extend([mols[i][0], mols[i][1]])
        pair_atoms.extend([[atoms[i]], [atoms[i]]])
        pair_pkas.extend([pkas[i], pkas[i]])
    return Draw.MolsToGridImage(draw_pairs, molsPerRow=2, subImgSize=(250, 250), highlightAtomLists=pair_atoms, legends=[f"pka = {pair_pkas[i]:.2f}" for i in range(12)])

def draw_sdf_mols(input_path, range_list = []):
    print(f"opening .sdf file at {input_path} and computing pkas...")
    with open(input_path, "rb") as fh:
        count = 0
        for i, mol in enumerate(Chem.ForwardSDMolSupplier(fh, removeHs=True)):
            if range_list and i not in range_list:
                continue
            props = mol.GetPropsAsDict()
            for prop in props.keys():
                mol.ClearProp(prop)
            mols, pkas, atoms = mol_query(mol)
            display(draw_pka_map(mols, pkas, atoms))
            print(f"Name: {mol.GetProp('_Name')}")