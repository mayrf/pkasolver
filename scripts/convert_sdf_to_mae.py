import os, subprocess

data_dir = "/data/local/"
schroedinger_dir = "/data/shared/software/schrodinger2021-1/"
perpare = f"{schroedinger_dir}/ligprep"
version = 0
sdf_file_name = f"mols_chembl_v{version}.sdf"
mae_file_name = f"mols_chembl_v{version}.mae"

# check that file is present
if not os.path.isfile(f"{data_dir}/{sdf_file_name}.gz"):
    raise RuntimeError(f"{data_dir}/{sdf_file_name}.gz file not found")

# convert to mae file
# http://gohom.win/ManualHom/Schrodinger/Schrodinger_2015-2_docs/ligprep/ligprep_user_manual.pdf
o = subprocess.run(
    [
        perpare,
        "-s 1",  # only one stereoisomer, if chiral tag not s et  choose R
        "-t 1",  # only most probable tautomer generated
        "-i 0",  # don't adjust the ionization state of the molecule
        "-isd",
        f"{data_dir}/{sdf_file_name}.gz",
        "-omae",
        f"{data_dir}/{mae_file_name}.gz",
    ],
    stderr=subprocess.STDOUT,
)
o.check_returncode()
