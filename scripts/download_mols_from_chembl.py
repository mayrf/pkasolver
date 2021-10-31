from chembl_webresource_client.new_client import new_client
from tqdm import tqdm
import gzip

molecule = new_client.molecule
molecule.set_format("sdf")
# mols = molecule.filter(max_phase=4)
mols = molecule.filter(molecule_type="Small molecule").filter(
    molecule_properties__num_ro5_violations=1
)
print(len(mols))

with gzip.open("/data/local/mols_chembl.sdf.gz", "wb+") as output:
    for mol in tqdm(mols):
        if mol:
            output.write(mol)
            output.write(b"$$$$\n")
