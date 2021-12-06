import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import ResonanceMolSupplier
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import tqdm

in_path = "/data/shared/projects/pkasolver-data/00_experimental_training_datasets.sdf"
out_path = "/data/shared/projects/pkasolver-data/misc_filtered_experimental_training_molecules.sdf"


def main():
    with Chem.SDWriter(out_path) as writer:
        with open(in_path, "rb") as fh:
            suppl = Chem.ForwardSDMolSupplier(fh, removeHs=True)
            i = 0
            for mol in tqdm.tqdm(suppl):
                props = mol.GetPropsAsDict()
                pka = props["pKa"]
                # pka = props["marvin_pKa"]
                pH = 7.4
                mol = Chem.RWMol(mol)
                atom = mol.GetAtomWithIdx(props["marvin_atom"])
                charge = atom.GetFormalCharge()
                Ex_Hs = atom.GetNumExplicitHs()
                Tot_Hs = atom.GetTotalNumHs()

                # make deprotonated conjugate as pKa > pH with at least one proton or
                # mol charge is positive (otherwise conjugate reaction center would have charge +2 --> highly unlikely)
                # if (pka > pH and Tot_Hs > 0) or charge > 0:
                #     atom.SetFormalCharge(charge - 1)
                #     if Ex_Hs > 0:
                #         atom.SetNumExplicitHs(Ex_Hs - 1)

                # make protonated conjugate as pKa < pH and charge is neutral or negative
                # elif pka <= pH and charge <= 0:
                #     atom.SetFormalCharge(charge + 1)
                #     if Tot_Hs == 0 or Ex_Hs > 0:
                #         atom.SetNumExplicitHs(Ex_Hs + 1)

                # make protonated conjugate as pKa > pH and there are no proton at the reaction center
                if (pka > pH and Tot_Hs == 0) or (pka < pH and charge > 0):
                    # print(f"wrong protonation")
                    writer.write(mol)
                    i += 1
        print(f"{i} wrongly protonated molecules filtered")
        # print(props)


if __name__ == "__main__":
    main()
