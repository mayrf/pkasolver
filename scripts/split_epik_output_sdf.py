from rdkit import Chem
import copy
import gzip

data_dir = "/data/shared/projects/pkasolver-data/"
schroedinger_dir = "/data/shared/software/schrodinger2021-1/"
# epik = schroedinger_dir + "epik"
# convert = schroedinger_dir + "utilities/structconvert"

input_file = f"{data_dir}chembl_epik.sdf"
suppl = Chem.SDMolSupplier(str(input_file), removeHs=False)
w = Chem.SDWriter(f"{data_dir}chembl_epik_split.sdf")
a = 0
for mol in suppl:
    try:
        props = mol.GetPropsAsDict()

        l = len([s for s in props.keys() if "r_epik_pKa" in s])

        for prop in props.keys():
            mol.ClearProp(prop)
        for i in range(l):
            mol_copy = copy.deepcopy(mol)
            mol_copy.SetProp(f"ID", props[f"chembl_id"])
            mol_copy.SetProp(f"pKa", props[f"r_epik_pKa_{i+1}"])
            mol_copy.SetProp(f"marvin_pKa", props[f"r_epik_pKa_{i+1}"])
            mol_copy.SetProp(f"marvin_atom", str(props[f"i_epik_pKa_atom_{i+1}"] - 1))
            mol_copy.SetProp(f"pka_number", str(i + 1))
            w.write(mol_copy)
            # print(props[f"r_epik_pKa_{i+1}"])
    except Exception as e:
        print(f"Error at molecule number {a}")
        print(e)
    a += 1
    # if a > 1000:
    #     break
w.close()
print(f"finished splitting {a} molecules")
