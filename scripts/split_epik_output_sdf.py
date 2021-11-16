from rdkit import Chem
from rdkit.Chem import Draw
import copy
import gzip
from pkasolver.chem import create_conjugate
from rdkit.Chem.AllChem import Compute2DCoords

version = 0
data_dir = "/data/shared/projects/pkasolver-data/"
schroedinger_dir = "/data/shared/software/schrodinger2021-1/"
input_file = f"{data_dir}chembl_epik.sdf"
suppl = Chem.SDMolSupplier(str(input_file), removeHs=True)
w = Chem.SDWriter(f"{data_dir}chembl_epik_split.sdf")
a = 0

for mol in suppl:
    Compute2DCoords(mol)
    try:
        props = mol.GetPropsAsDict()
        l = len([s for s in props.keys() if "r_epik_pKa" in s])
        atoms, pkas = [], []
        for i in range(l):
            pkas.append(
                (
                    float(props[f"r_epik_pKa_{i+1}"]),
                    int(props[f"i_epik_pKa_atom_{i+1}"]) - 1,
                    props[f"chembl_id"],
                )
            )
        acids = [pka for pka in pkas if pka[0] < 7]
        bases = pkas[len(acids) :]
        for prop in props.keys():
            mol.ClearProp(prop)
        acid_mols = [mol]
        for i, acid in enumerate(acids[::-1]):
            new_mol = create_conjugate(acid_mols[i], acid[1], acid[0], pH=7)
            new_mol.SetProp(f"ID", str(acid[2]))
            new_mol.SetProp(f"pKa", str(acid[0]))
            new_mol.SetProp(f"marvin_pKa", str(acid[0]))
            new_mol.SetProp(f"marvin_atom", str(acid[1]))
            new_mol.SetProp(f"pka_number", f"acid_{i + 1}")
            acid_mols.append(new_mol)
        base_mols = [mol]
        for i, base in enumerate(bases):
            new_mol = create_conjugate(base_mols[i], base[1], base[0], pH=7)
            new_mol.SetProp(f"ID", str(base[2]))
            new_mol.SetProp(f"pKa", str(base[0]))
            new_mol.SetProp(f"marvin_pKa", str(base[0]))
            new_mol.SetProp(f"marvin_atom", str(base[1]))
            new_mol.SetProp(f"pka_number", f"base_{i + 1}")
            base_mols.append(new_mol)
        mols = acid_mols[1:] + base_mols[1:]
        for mol in mols:
            w.write(mol)

    except Exception as e:
        print(f"Error at molecule number {a}")
        print(e)
        Draw.MolToFile(mol, f"{a}.svg", size=(600, 600), addAtomIndices=True)
    a += 1
    if a > 1000:
        break
w.close()
print(f"finished splitting {a} molecules")
