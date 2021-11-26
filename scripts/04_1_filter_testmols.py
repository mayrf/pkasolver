import argparse
import gzip
from rdkit import Chem
import tqdm
from rdkit import RDLogger


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
                ini_list.extend(ini_filter(suppl, args))
            with gzip.open(i, "r") as fh:
                suppl = Chem.ForwardSDMolSupplier(fh, removeHs=False)
                smi_list.extend(smi_filter(suppl, args))
        else:
            with open(i, "rb") as fh:
                suppl = Chem.ForwardSDMolSupplier(fh, removeHs=False)
                ini_list.extend(ini_filter(suppl, args))
            with open(i, "rb") as fh:
                suppl = Chem.ForwardSDMolSupplier(fh, removeHs=False)
                smi_list.extend(smi_filter(suppl, args))

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


def smi_filter(suppl, args):
    smi_list = []
    for mol in suppl:
        smi_list.append(Chem.MolToSmiles(mol))
    # print(smi_list)
    return smi_list


def ini_filter(suppl, args):
    ini_list = []
    for mol in suppl:
        ini_list.append(Chem.inchi.MolToInchi(mol))
    return ini_list


def processing(suppl, args, ini_list, smi_list):
    dup = 0
    with Chem.SDWriter(args.output) as writer:
        for mol in tqdm.tqdm(suppl):

            if Chem.inchi.MolToInchi(mol) in ini_list:
                if Chem.MolToSmiles(mol) in smi_list:
                    print(" inchi and smiles duplicate found!")
                else:
                    print(" inchi duplicate found!")
                dup += 1
            elif Chem.MolToSmiles(mol) in smi_list:
                print("smiles duplicate found!")
                dup += 1
            else:
                writer.write(mol)
    print(f"{dup} duplicate molecules found and discarted")
    # pass


if __name__ == "__main__":
    main()
