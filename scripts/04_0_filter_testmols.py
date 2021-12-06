import argparse
import gzip
from rdkit import Chem
import tqdm
from rdkit import RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize


def main():
    RDLogger.DisableLog("rdApp.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input filename")
    parser.add_argument("--filter", help="filter filename")
    parser.add_argument("--output", help="output filename")
    args = parser.parse_args()
    input_zipped = False
    print("inputfile:", args.input)
    print("outputfile:", args.output)
    ini_list = []
    smi_list = []
    for i in args.filter.split(","):
        # test if it's gzipped
        with gzip.open(i, "r") as fh:
            try:
                fh.read(1)
                input_zipped = True
            except gzip.BadGzipFile:
                input_zipped = False

        if input_zipped:
            with gzip.open(i, "r") as fh:
                suppl = Chem.ForwardSDMolSupplier(fh, removeHs=False)
                ini_list.extend(ini_filter(suppl))
            with gzip.open(i, "r") as fh:
                suppl = Chem.ForwardSDMolSupplier(fh, removeHs=False)
                smi_list.extend(smi_filter(suppl))
        else:
            with open(i, "rb") as fh:
                suppl = Chem.ForwardSDMolSupplier(fh, removeHs=False)
                ini_list.extend(ini_filter(suppl))
            with open(i, "rb") as fh:
                suppl = Chem.ForwardSDMolSupplier(fh, removeHs=False)
                smi_list.extend(smi_filter(suppl))

    print(f"{len(ini_list)} inchi test molecules found")
    print(f"{len(smi_list)} smi test molecules found")
    # test if it's gzipped
    with gzip.open(args.input, "r") as fh:
        try:
            fh.read(1)
            input_zipped = True
        except gzip.BadGzipFile:
            input_zipped = False

    if input_zipped:
        with gzip.open(args.input, "r") as fh:
            suppl = Chem.ForwardSDMolSupplier(fh, removeHs=True)
            processing(suppl, args, ini_list, smi_list)
    else:
        with open(args.input, "rb") as fh:
            suppl = Chem.ForwardSDMolSupplier(fh, removeHs=True)
            processing(suppl, args, ini_list, smi_list)


def smi_filter(suppl):
    un = rdMolStandardize.Uncharger()
    smi_list = []
    for mol in suppl:
        mol = un.uncharge(mol)
        smi_list.append(Chem.MolToSmiles(mol))
    return smi_list


def ini_filter(suppl):
    un = rdMolStandardize.Uncharger()
    ini_list = []
    for mol in suppl:
        mol = un.uncharge(mol)
        ini_list.append(Chem.inchi.MolToInchi(mol))
    return ini_list


def processing(suppl, args, ini_list, smi_list):
    dup = 0
    skipped = 0
    written = 0
    un = rdMolStandardize.Uncharger()
    with gzip.open(args.output, "wt+") as sdf_zip:
        with Chem.SDWriter(sdf_zip) as writer:
            for idx, mol in enumerate(tqdm.tqdm(suppl)):
                if mol:
                    mol_uncharged = un.uncharge(mol)
                    inchi = Chem.inchi.MolToInchi(mol_uncharged)
                    smiles = Chem.MolToSmiles(mol_uncharged)
                    if inchi in ini_list or smiles in smi_list:
                        dup += 1
                    else:
                        written += 1
                        writer.write(mol)

                else:
                    skipped += 1

    print(f"{dup} duplicate molecules found and discarted")
    print(f"{skipped} molecules skipped")
    print(f"{written} molecules")


if __name__ == "__main__":
    main()
