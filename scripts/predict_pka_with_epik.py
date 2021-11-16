import os, subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input filename")
parser.add_argument("--output", help="output filename")
args = parser.parse_args()

print("inputfile:", args.input)
print("outputfile:", args.output)

mae_file_name = args.input
mae_file_name_with_pka = args.output

schroedinger_dir = "/data/shared/software/schrodinger2021-1/"
epik = f"{schroedinger_dir}/epik"

if not os.path.isfile(mae_file_name):
    raise RuntimeError(f"{mae_file_name} file not found")

# predict pka of mols in .mae files with epik
o = subprocess.run(
    [
        f"{epik}",
        "-scan",
        "-imae",
        mae_file_name,
        "-omae",
        mae_file_name_with_pka,
        "-p",
        "0.1",
    ]
)
o.check_returncode()
