import os, subprocess

version = 1
mae_file_name = f"mols_chembl_v{version}.mae"
mae_file_name_with_pka = f"mols_chembl_with_pka_for_v{version}.mae"

data_dir = "/data/local/"
schroedinger_dir = "/data/shared/software/schrodinger2021-1/"
epik = f"{schroedinger_dir}/epik"

if not os.path.isfile(f"{data_dir}/{mae_file_name}"):
    raise RuntimeError(f"{data_dir}/{mae_file_name} file not found")

# predict pka of mols in .mae files with epik
o = subprocess.run(
    [
        f"{epik}",
        "-scan",
        "-imae",
        f"{data_dir}/{mae_file_name}",
        "-omae",
        f"{data_dir}/{mae_file_name_with_pka}",
    ]
)
o.check_returncode()
