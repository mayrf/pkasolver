import os, subprocess

data_dir = "/data/shared/projects/pkasolver-data"
schroedinger_dir = "/data/shared/software/schrodinger2021-1/"
convert = f"{schroedinger_dir}/utilities/sdconvert"
version = 0
sdf_file_name = f"mols_chembl_with_pka_for_v{version}.sdf"
mae_file_name = f"mols_chembl_with_pka_for_v{version}.mae"

# check that file is present
if not os.path.isfile(f"{data_dir}/{mae_file_name}.gz"):
    raise RuntimeError(f"{data_dir}/{mae_file_name}.gz file not found")

# convert to mae file
# http://gohom.win/ManualHom/Schrodinger/Schrodinger_2015-2_docs/ligprep/ligprep_user_manual.pdf
o = subprocess.run(
    [
        convert,
        "-imae",
        "{data_dir}/{mae_file_name}.gz",
        "-osdf",
        f"{data_dir}/{sdf_file_name}.gz",
        "-annstereo",
        "-pKa",
    ],
    stderr=subprocess.STDOUT,
)
o.check_returncode()
