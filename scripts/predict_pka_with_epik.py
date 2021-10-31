import os, subprocess

data_dir = "/data/local/"
schroedinger_dir = "/data/shared/software/schrodinger2021-1/"
epik = f"{schroedinger_dir}/epik"

if not os.path.isfile(f"{data_dir}/mols_chembl.mae"):
    raise RuntimeError(f"{data_dir}/mols_chembl.mae file not found")

# predict pka of mols in .mae files with epik
o = subprocess.run(
    [
        f"{epik}",
        "-scan",
        "-imae",
        f"{data_dir}/mols_chembl.mae",
        "-omae",
        f"{data_dir}/mols_chembl_epik.mae",
    ]
)
o.check_returncode()
