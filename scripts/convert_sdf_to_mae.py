import os, subprocess
import fileinput, gzip

data_dir = "/data/local/"
schroedinger_dir = "/data/shared/software/schrodinger2021-1/"
convert = f"{schroedinger_dir}/utilities/structconvert"
version = 1
sdf_file_name = f"mols_chembl_v{version}.sdf"
mae_file_name = f"mols_chembl_v{version}.mae"

# unzip file and copy to working directory
with gzip.open(f"{data_dir}/{sdf_file_name}.gz", "rb+") as input:
    with open(f"{data_dir}/{sdf_file_name}", "w+") as output:
        for line in input:
            output.write(line.decode())

# check that file is present
if not os.path.isfile(f"{data_dir}/{sdf_file_name}"):
    raise RuntimeError(f"{data_dir}/{sdf_file_name} file not found")

# correct for possible double '$$$$' pattern
last_line = ""
for line in fileinput.input(f"{data_dir}/{sdf_file_name}", inplace=True):
    if line == last_line:
        continue
    else:
        last_line = line
        print(line, end="")

# convert to mae file
o = subprocess.run(
    [convert, f"{data_dir}/{sdf_file_name}", f"{data_dir}/{mae_file_name}"],
    stderr=subprocess.STDOUT,
)
o.check_returncode()
