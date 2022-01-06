import argparse
import os
import subprocess


def main():
    """
    takes a sdf file (can be gzipped) and converts it to a Schrödinger maestro (mae) file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input filename, type: .sdf.gz or .sdf")
    parser.add_argument("--output", help="output filename, type: .mae.gz or .mae")
    args = parser.parse_args()

    print("inputfile:", args.input)
    print("outputfile:", args.output)

    sdf_file_name = args.input
    mae_file_name = args.output

    schroedinger_dir = "/data/shared/software/schrodinger2021-1/"
    ligprep = f"{schroedinger_dir}/ligprep"

    # check that file is present
    if not os.path.isfile(f"{sdf_file_name}"):
        raise RuntimeError(f"{sdf_file_name} file not found")

    # convert to mae file
    # http://gohom.win/ManualHom/Schrodinger/Schrodinger_2015-2_docs/ligprep/ligprep_user_manual.pdf
    o = subprocess.run(
        [
            ligprep,
            "-s 1",  # only one stereoisomer, if chiral tag not s et  choose R
            "-t 1",  # only most probable tautomer generated
            "-i 0",  # don't adjust the ionization state of the molecule
            "-isd",
            sdf_file_name,
            "-omae",
            mae_file_name,
            "-WAIT",
        ],
        stderr=subprocess.STDOUT,
    )
    o.check_returncode()


if __name__ == "__main__":
    main()
