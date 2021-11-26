from chembl_webresource_client.new_client import new_client
from tqdm import tqdm
import gzip
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="output filename")
    args = parser.parse_args()

    print("outputfile:", args.output)

    molecule = new_client.molecule
    # mols = molecule.filter(max_phase=4)
    mols = molecule.filter(molecule_type="Small molecule").filter(
        molecule_properties__num_ro5_violations=1
    )
    print(len(mols))

    with gzip.open(args.output, "wb+") as output:
        for mol in tqdm(mols):
            if mol["molecule_structures"]:
                output.write(mol["molecule_structures"]["molfile"].encode())
                output.write(b"$$$$\n")


if __name__ == "__main__":
    main()
