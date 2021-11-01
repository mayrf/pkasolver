from chembl_webresource_client.new_client import new_client
from tqdm import tqdm
import gzip

version = 1
sdf_file_name = f"mols_chembl_v{version}.sdf"

molecule = new_client.molecule
# mols = molecule.filter(max_phase=4)
mols = molecule.filter(molecule_type="Small molecule").filter(
    molecule_properties__mw_freebase__gte=70
)
print(len(mols))

with gzip.open(f"/data/local/{sdf_file_name}.gz", "wb+") as output:
    for mol in tqdm(mols):
        if mol["molecule_structures"]:
            output.write(mol["molecule_structures"]["molfile"].encode())
            output.write(b"$$$$\n")
