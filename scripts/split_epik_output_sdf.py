from rdkit import Chem
import copy
import gzip

version = 0
data_dir = "/data/shared/projects/pkasolver-data/"
schroedinger_dir = "/data/shared/software/schrodinger2021-1/"

input_file = gzip.open(f"{data_dir}/mols_chembl_with_pka_for_v{version}.sdf.gz", "rb")
output_file = gzip.open(
    f"{data_dir}/split_mols_chembl_with_pka_for_v{version}.sdf.gz", "wt+"
)

reader = Chem.ForwardSDMolSupplier(input_file, removeHs=False)
writer = Chem.SDWriter(output_file)


for idx, mol in enumerate(reader):

    try:
        props = mol.GetPropsAsDict()
    except AttributeError as e:
        print(e)
        continue

    if not any("r_epik_pKa" in string for string in props.keys()):
        # skip if no pka value was calculated
        print(f"No pKa set for mol {idx} ...")
        continue

    # calculate number of pka values set
    number_of_pka_values = len([s for s in props.keys() if "r_epik_pKa" in s])

    # clear poperty entries
    for prop in props.keys():
        mol.ClearProp(prop)

    # for each calculated pka value
    for i in range(number_of_pka_values):
        mol_copy = copy.deepcopy(mol)
        mol_copy.SetProp(f"ID", props[f"chembl_id"])
        mol_copy.SetProp(f"pKa", props[f"r_epik_pKa_{i+1}"])
        mol_copy.SetProp(f"marvin_pKa", props[f"r_epik_pKa_{i+1}"])
        mol_copy.SetProp(f"marvin_atom", str(props[f"i_epik_pKa_atom_{i+1}"] - 1))
        mol_copy.SetProp(f"pka_number", str(i + 1))
        writer.write(mol_copy)

writer.close()
print(f"finished splitting {idx} molecules")
