from rdkit import Chem
from pkasolver.chem import create_conjugate
import argparse
import gzip
from molvs import Standardizer
from copy import deepcopy
import pickle
from rdkit.Chem import PropertyMol

s = Standardizer()

PH = 7.4


def main():
    """
    takes sdf file with molcules containing epik pka predictions in their properties
    and outputs a new sdf where those molecules containing more than one pka
    get duplicated so that every molecules only contains one pka value.
    the molecule associated with each pka is the protonted form of the respective pka reaction
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input filename")
    parser.add_argument("--output", help="output filename")
    args = parser.parse_args()
    input_zipped = False
    print(f"pH splitting used: {PH}")
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


# generating the datat for a single molecule for all
# acidic pKa values
def iterate_over_acids(
    acidic_mols_properties,
    nr_of_mols,
    partner_mol: Chem.Mol,
    nr_of_skipped_mols,
    pka_list: list,
    GLOBAL_COUNTER: int,
    counter_list: list,
    smiles_list: list,
):

    acidic_mols = []
    skipping_acids = 0

    for idx, acid_prop in enumerate(
        reversed(acidic_mols_properties)
    ):  # list must be iterated in reverse, in order to protonated the strongest conjugate base first

        if skipping_acids == 0:  # if a acid was skipped, all further acids are skipped
            try:
                new_mol = create_conjugate(
                    partner_mol,
                    acid_prop["atom_idx"],
                    acid_prop["pka_value"],
                    pH=PH,
                )
                Chem.SanitizeMol(new_mol)
                # new_mol = s.standardize(new_mol)

            except Exception as e:
                print(f"Error at molecule number {nr_of_mols} - acid enumeration")
                print(e)
                print(acid_prop)
                print(acidic_mols_properties)
                if partner_mol:
                    print(Chem.MolToSmiles(partner_mol))
                skipping_acids += 1
                nr_of_skipped_mols += 1
                continue  # continue instead of break, will not enter this routine gain since skipping_acids != 0

            pka_list.append(acid_prop["pka_value"])
            smiles_list.append(
                (Chem.MolToSmiles(new_mol), Chem.MolToSmiles(partner_mol))
            )

            for mol in [new_mol, partner_mol]:
                GLOBAL_COUNTER += 1
                counter_list.append(GLOBAL_COUNTER)
                mol.SetProp(f"CHEMBL_ID", str(acid_prop["chembl_id"]))
                mol.SetProp(f"INTERNAL_ID", str(GLOBAL_COUNTER))
                mol.SetProp(f"pKa", str(acid_prop["pka_value"]))
                mol.SetProp(f"epik_atom", str(acid_prop["atom_idx"]))
                mol.SetProp(f"pKa_number", f"acid_{idx + 1}")
                mol.SetProp(f"mol-smiles", f"{Chem.MolToSmiles(mol)}")

            # add current mol to list of acidic mol. for next
            # lower pKa value, this mol is starting structure
            acidic_mols.append(
                (
                    PropertyMol.PropertyMol(new_mol),
                    PropertyMol.PropertyMol(partner_mol),
                )
            )
            partner_mol = deepcopy(new_mol)

        else:
            skipping_acids += 1
    return acidic_mols, nr_of_skipped_mols, GLOBAL_COUNTER, skipping_acids


def iterate_over_bases(
    basic_mols_properties,
    nr_of_mols,
    partner_mol: Chem.Mol,
    nr_of_skipped_mols,
    pka_list: list,
    GLOBAL_COUNTER: int,
    counter_list: list,
    smiles_list: list,
):

    basic_mols = []
    skipping_bases = 0
    for idx, basic_prop in enumerate(basic_mols_properties):
        if skipping_bases == 0:  # if a base was skipped, all further bases are skipped
            try:
                new_mol = create_conjugate(
                    partner_mol,
                    basic_prop["atom_idx"],
                    basic_prop["pka_value"],
                    pH=PH,
                )

                Chem.SanitizeMol(new_mol)
                # new_mol = s.standardize(new_mol)

            except Exception as e:
                # in case error occurs new_mol is not in basic list
                print(f"Error at molecule number {nr_of_mols} - bases enumeration")
                print(e)
                print(basic_prop)
                print(basic_mols_properties)
                if partner_mol:
                    print(Chem.MolToSmiles(partner_mol))
                skipping_bases += 1
                nr_of_skipped_mols += 1
                continue

            pka_list.append(basic_prop["pka_value"])
            smiles_list.append(
                (Chem.MolToSmiles(partner_mol), Chem.MolToSmiles(new_mol))
            )

            for mol in [partner_mol, new_mol]:
                GLOBAL_COUNTER += 1
                counter_list.append(GLOBAL_COUNTER)
                mol.SetProp(f"CHEMBL_ID", str(basic_prop["chembl_id"]))
                mol.SetProp(f"INTERNAL_ID", str(GLOBAL_COUNTER))
                mol.SetProp(f"pKa", str(basic_prop["pka_value"]))
                mol.SetProp(f"epik_atom", str(basic_prop["atom_idx"]))
                mol.SetProp(f"pKa_number", f"acid_{idx + 1}")
                mol.SetProp(f"mol-smiles", f"{Chem.MolToSmiles(mol)}")

            # add current mol to list of acidic mol. for next
            # lower pKa value, this mol is starting structure
            basic_mols.append(
                (PropertyMol.PropertyMol(partner_mol), PropertyMol.PropertyMol(new_mol))
            )
            partner_mol = deepcopy(new_mol)

        else:
            skipping_bases += 1

    return basic_mols, nr_of_skipped_mols, GLOBAL_COUNTER, skipping_bases


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

        print(pkas)

        # calculate number of acidic and basic pka values
        upper_pka_limit = 16
        lower_pka_limit = -2
        nr_of_acids = sum(
            pka["pka_value"] <= PH and pka["pka_value"] > lower_pka_limit
            for pka in pkas
        )
        nr_of_bases = sum(
            pka["pka_value"] > PH and pka["pka_value"] < upper_pka_limit for pka in pkas
        )
        assert nr_of_acids + nr_of_bases <= len(pkas)

        acidic_mols_properties = [
            mol_pka
            for mol_pka in pkas
            if mol_pka["pka_value"] <= PH and mol_pka["pka_value"] > lower_pka_limit
        ]
        basic_mols_properties = [
            mol_pka
            for mol_pka in pkas
            if mol_pka["pka_value"] > PH and mol_pka["pka_value"] < upper_pka_limit
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
        mol_at_ph7 = mol
        print(Chem.MolToSmiles(mol_at_ph7))
        acidic_mols = []
        partner_mol = deepcopy(mol_at_ph7)
        (
            acidic_mols,
            nr_of_skipped_mols,
            GLOBAL_COUNTER,
            skipping_acids,
        ) = iterate_over_acids(
            acidic_mols_properties,
            nr_of_mols,
            partner_mol,
            nr_of_skipped_mols,
            pka_list,
            GLOBAL_COUNTER,
            counter_list,
            smiles_list,
        )

        # same workflow for basic mols
        basic_mols = []
        partner_mol = deepcopy(mol_at_ph7)
        (
            basic_mols,
            nr_of_skipped_mols,
            GLOBAL_COUNTER,
            skipping_bases,
        ) = iterate_over_bases(
            basic_mols_properties,
            nr_of_mols,
            partner_mol,
            nr_of_skipped_mols,
            pka_list,
            GLOBAL_COUNTER,
            counter_list,
            smiles_list,
        )

        # combine basic and acidic mols, skip neutral mol for acids
        combined_mols = acidic_mols + basic_mols
        if (
            len(combined_mols)
            != len(acidic_mols_properties)
            - skipping_acids
            + len(basic_mols_properties)
            - skipping_bases
        ):
            raise RuntimeError(
                combined_mols,
                acidic_mols_properties,
                skipping_acids,
                basic_mols_properties,
                skipping_bases,
            )

        if len(combined_mols) != 0:
            chembl_id = combined_mols[0][0].GetProp("CHEMBL_ID")
            print(f"CHEMBL_ID: {chembl_id}")
            for mols in combined_mols:
                if mols[0].GetProp("pKa") != mols[1].GetProp("pKa"):
                    raise AssertionError(mol[0].GetProp("pKa"), mol[1].GetProp("pKa"))

                mol1, mol2 = mols
                pka = mol1.GetProp("pKa")
                counter = mol1.GetProp("INTERNAL_ID")
                print(
                    f"{counter=}, {pka=}, {mol1.GetProp('mol-smiles')}, prot, {mol1.GetProp('epik_atom')}"
                )
                pka = mol2.GetProp("pKa")
                counter = mol2.GetProp("INTERNAL_ID")
                print(
                    f"{counter=}, {pka=}, {mol2.GetProp('mol-smiles')}, deprot, {mol1.GetProp('epik_atom')}"
                )
            print(pka_list)
            if chembl_id in all_protonation_states_enumerated.keys():
                raise RuntimeError("Repeated chembl id!")

            all_protonation_states_enumerated[chembl_id] = {
                "mols": combined_mols,
                "pKa_list": pka_list,
                "smiles_list": smiles_list,
                "counter_list": counter_list,
            }

    print(f"finished splitting {nr_of_mols} molecules")
    print(f"skipped mols: {nr_of_skipped_mols}")
    pickle.dump(all_protonation_states_enumerated, open(args.output, "wb+"))


if __name__ == "__main__":
    main()
