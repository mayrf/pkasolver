import argparse
import pickle

from pkasolver import run_with_mol_list
from rdkit import Chem

parser = argparse.ArgumentParser()
parser.add_argument("--smiles", help="training set filename, type: .pkl")
parser.add_argument("--min_ph", help="training set filename, type: .pkl")
parser.add_argument("--max_ph", help="training set filename, type: .pkl")
parser.add_argument("--pka_precision", help="training set filename, type: .pkl")
args = parser.parse_args()

mol = Chem.MolFromSmiles(args.smiles)
print(args.smiles)
print(mol)
mol = run_with_mol_list(
    [mol],
    min_ph=float(args.min_ph),
    max_ph=float(args.max_ph),
    pka_precision=float(args.pka_precision),
)
pickle.dump(mol, open("test.pkl", "wb"))

