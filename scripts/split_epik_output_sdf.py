from rdkit import Chem
from pkasolver.chem import create_conjugate
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input filename")
parser.add_argument("--output", help="output filename")
args = parser.parse_args()

print("inputfile:", args.input)
print("outputfile:", args.output)


suppl = Chem.SDMolSupplier(str(args.input), removeHs=True)

with Chem.SDWriter(args.output) as writer:
    for nr_of_mols, mol in enumerate(suppl):
        props = mol.GetPropsAsDict()
        nr_of_protonation_states = len([s for s in props.keys() if "r_epik_pKa" in s])
        pkas = []
        for i in range(nr_of_protonation_states):
            pkas.append(
                (
                    float(props[f"r_epik_pKa_{i+1}"]),
                    int(props[f"i_epik_pKa_atom_{i+1}"]) - 1,
                    props[f"chembl_id"],
                )
            )

        # calculate number of acidic and basic pka values
        nr_of_acids = sum(pka[0] <= 7 for pka in pkas)
        nr_of_bases = sum(pka[0] > 7 for pka in pkas)
        assert nr_of_acids + nr_of_bases == len(pkas)

        acidic_mols_properties = [mol_prop for mol_prop in pkas if mol[0] <= 7]
        basic_mols_properties = [mol_prop for mol_prop in pkas if mol[0] > 7]

        assert len(acidic_mols_properties) == nr_of_acids
        assert len(basic_mols_properties) == nr_of_bases

        # clear porps
        for prop in props.keys():
            mol.ClearProp(prop)

        # add neutral mol as first acidic mol
        acidic_mols = [mol]
        for idx, acid_prop in enumerate(acidic_mols_properties):
            try:
                new_mol = create_conjugate(
                    acidic_mols[-1], acid_prop[1], acid_prop[0], pH=7
                )
                # Chem.SanitizeMol(new_mol)

            except Exception as e:
                print(f"Error at molecule number {nr_of_mols}")
                print(e)

            new_mol.SetProp(f"ID", str(acid_prop[2]))
            new_mol.SetProp(f"pKa", str(acid_prop[0]))
            new_mol.SetProp(f"marvin_pKa", str(acid_prop[0]))
            new_mol.SetProp(f"marvin_atom", str(acid_prop[1]))
            new_mol.SetProp(f"pka_number", f"acid_{idx + 1}")
            # add current mol to list of acidic mol. for next
            # lower pKa value, this mol is starting structure
            acidic_mols.append(new_mol)

        # same workflow for basic mols
        basic_mols = [mol]
        for idx, basic_prop in enumerate(basic_mols_properties):
            try:
                new_mol = create_conjugate(
                    basic_mols[-1], basic_prop[1], basic_prop[0], pH=7
                )
                # Chem.SanitizeMol(new_mol)
            except Exception as e:
                print(f"Error at molecule number {nr_of_mols}")
                print(e)

            new_mol.SetProp(f"ID", str(basic_prop[2]))
            new_mol.SetProp(f"pKa", str(basic_prop[0]))
            new_mol.SetProp(f"marvin_pKa", str(basic_prop[0]))
            new_mol.SetProp(f"marvin_atom", str(basic_prop[1]))
            new_mol.SetProp(f"pka_number", f"base_{idx + 1}")
            basic_mols.append(new_mol)

        # combine basic and acidic mols, skipp neutral mol
        mols = acidic_mols[1:] + basic_mols[1:]
        assert len(mols) == len(acidic_mols_properties) + len(basic_mols_properties)

        for mol in mols:
            writer.write(mol)

print(f"finished splitting {nr_of_mols} molecules")
