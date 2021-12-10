from rdkit import Chem
from pkasolver.chem import create_conjugate
import argparse
import gzip

PH = 7.4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input filename")
    parser.add_argument("--output", help="output filename")
    args = parser.parse_args()
    input_zipped = False
    print("inputfile:", args.input)
    print("outputfile:", args.output)

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
            processing(suppl, args)
    else:
        with open(args.input, "rb") as fh:
            suppl = Chem.ForwardSDMolSupplier(fh, removeHs=True)
            processing(suppl, args)


def processing(suppl, args):
    GLOBAL_COUNTER = 0
    nr_of_skipped_mols = 0
    all_protonation_states_enumerated = dict()
    for nr_of_mols, mol in enumerate(suppl):

        if not mol:
            continue

        skipping_bases = 0
        skipping_acids = 0

        try:
            props = mol.GetPropsAsDict()
        except AttributeError as e:
            # this mol has no pka value
            nr_of_skipped_mols += 1
            print(e)
            continue

        nr_of_protonation_states = len([s for s in props.keys() if "r_epik_pKa" in s])

        pkas = []

        for i in range(nr_of_protonation_states):
            pkas.append(
                {
                    "pka_value": float(props[f"r_epik_pKa_{i+1}"]),
                    "atom_idx": int(props[f"i_epik_pKa_atom_{i+1}"]) - 1,
                    "chembl_id": props[f"chembl_id"],
                }
            )

        # calculate number of acidic and basic pka values
        nr_of_acids = sum(
            pka["pka_value"] <= PH and pka["pka_value"] > 0.5 for pka in pkas
        )
        nr_of_bases = sum(
            pka["pka_value"] > PH and pka["pka_value"] < 13.5 for pka in pkas
        )
        assert nr_of_acids + nr_of_bases <= len(pkas)

        acidic_mols_properties = [
            mol_pka
            for mol_pka in pkas
            if mol_pka["pka_value"] <= PH and mol_pka["pka_value"] > 0.5
        ]
        basic_mols_properties = [
            mol_pka
            for mol_pka in pkas
            if mol_pka["pka_value"] > PH and mol_pka["pka_value"] < 13.5
        ]

        if len(acidic_mols_properties) != nr_of_acids:
            raise RuntimeError(f"{acidic_mols_properties=}, {nr_of_acids=}")
        if len(basic_mols_properties) != nr_of_bases:
            raise RuntimeError(f"{basic_mols_properties=}, {nr_of_bases=}")

        # clear porps
        for prop in props.keys():
            mol.ClearProp(prop)

        # save values
        pka_list = []
        smiles_list = []
        counter_list = []

        # add mol at pH=7.4
        acidic_mols = [mol]
        for idx, acid_prop in enumerate(
            reversed(acidic_mols_properties)
        ):  # list must be iterated in reverse, in order to protonated the strongest conjugate base first
            if (
                skipping_acids == 0
            ):  # if a acid was skipped, all further acids are skipped
                try:
                    new_mol = create_conjugate(
                        acidic_mols[-1],
                        acid_prop["atom_idx"],
                        acid_prop["pka_value"],
                        pH=PH,
                    )
                    Chem.SanitizeMol(new_mol)
                except Exception as e:
                    print(f"Error at molecule number {nr_of_mols}")
                    print(e)
                    print(acid_prop)
                    print(acidic_mols_properties)
                    print(Chem.MolToMolBlock(mol))
                    skipping_acids += 1
                    nr_of_skipped_mols += 1
                    continue  # continue instead of break, will not enter this routine gain since skipping_acids != 0

                GLOBAL_COUNTER += 1
                pka_list.append(acid_prop["pka_value"])
                smiles_list.append(Chem.MolToSmiles(new_mol))
                counter_list.append(GLOBAL_COUNTER)
                new_mol.SetProp(f"CHEMBL_ID", str(acid_prop["chembl_id"]))
                new_mol.SetProp(f"INTERNAL_ID", str(GLOBAL_COUNTER))
                new_mol.SetProp(f"pKa", str(acid_prop["pka_value"]))
                new_mol.SetProp(f"epik_atom", str(acid_prop["atom_idx"]))
                new_mol.SetProp(f"pKa_number", f"acid_{idx + 1}")
                new_mol.SetProp(f"mol-smiles", f"{Chem.MolToSmiles(new_mol)}")

                # add current mol to list of acidic mol. for next
                # lower pKa value, this mol is starting structure
                acidic_mols.append(new_mol)

            else:
                skipping_acids += 1

        # same workflow for basic mols
        basic_mols = [mol]
        for idx, basic_prop in enumerate(basic_mols_properties):

            if idx == 0:  # special case for mol at pH=7.4
                GLOBAL_COUNTER += 1
                new_mol = basic_mols[-1]
                new_mol.SetProp(f"CHEMBL_ID", str(basic_prop["chembl_id"]))
                new_mol.SetProp(f"INTERNAL_ID", str(GLOBAL_COUNTER))
                new_mol.SetProp(f"epik_atom", str(basic_prop["atom_idx"]))
                new_mol.SetProp(f"pKa", "NEUTRAL")
                new_mol.SetProp(f"mol-smiles", f"{Chem.MolToSmiles(new_mol)}")

                pka_list.append(basic_prop["pka_value"])
                smiles_list.append(Chem.MolToSmiles(new_mol))
                counter_list.append(GLOBAL_COUNTER)

            if (
                skipping_bases == 0
            ):  # if a base was skipped, all further bases are skipped
                try:

                    basic_mols.append(
                        create_conjugate(
                            new_mol,
                            basic_prop["atom_idx"],
                            basic_prop["pka_value"],
                            pH=PH,
                        )
                    )
                    Chem.SanitizeMol(basic_mols[-1])

                except Exception as e:
                    # in case error occurs new_mol is not in basic list
                    print(f"Error at molecule number {nr_of_mols}")
                    print(e)
                    print(basic_prop)
                    print(basic_mols_properties)
                    print(Chem.MolToMolBlock(mol))
                    skipping_bases += 1
                    nr_of_skipped_mols += 1
                    continue

                new_mol = basic_mols[-1]
                GLOBAL_COUNTER += 1
                new_mol.SetProp(f"CHEMBL_ID", str(basic_prop["chembl_id"]))

                new_mol.SetProp(f"INTERNAL_ID", str(GLOBAL_COUNTER))
                new_mol.SetProp(f"pKa", str(basic_prop["pka_value"]))

                new_mol.SetProp(f"epik_atom", str(basic_prop["atom_idx"]))
                new_mol.SetProp(f"pKa_number", f"acid_{idx + 1}")
                new_mol.SetProp(f"mol-smiles", f"{Chem.MolToSmiles(new_mol)}")

                pka_list.append(basic_prop["pka_value"])
                smiles_list.append(Chem.MolToSmiles(new_mol))
                counter_list.append(GLOBAL_COUNTER)

            else:
                skipping_bases += 1

        # prepare last basic mol, which do

        # combine basic and acidic mols, skip neutral mol for acids
        mols = acidic_mols[1:] + basic_mols
        assert (
            len(mols)
            == len(acidic_mols_properties)
            - skipping_acids
            + len(basic_mols_properties)
            + 1  # because we add the last protonation state
            - skipping_bases
        )

        if len(mols) != 0:
            chembl_id = mols[-1].GetProp("CHEMBL_ID")
            print(f"CHEMBL_ID: {chembl_id}")

            for mol in mols:
                pka = mol.GetProp("pKa")
                counter = mol.GetProp("INTERNAL_ID")
                mol_smiles = mol.GetProp("mol-smiles")
                print(f"{counter=}, {pka=}, {mol_smiles}")

            if chembl_id in all_protonation_states_enumerated.keys():
                raise RuntimeError("Repreated chembl id!")

            all_protonation_states_enumerated[chembl_id] = {
                "mols": mols,
                "pKa_list": pka_list,
                "smiles_list": smiles_list,
                "counter_list": counter_list,
            }

    print(f"finished splitting {nr_of_mols} molecules")
    print(f"skipped mols: {nr_of_skipped_mols}")


if __name__ == "__main__":
    main()
