import os, subprocess
import argparse


def main():
    """
    takes Schrödinger maestro (mae) file (can be gzipped) and converts it to sdf file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input filename, type: .mae.gz or .mae")
    parser.add_argument("--output", help="output filename, type: .sdf.gz or .sdf")
    args = parser.parse_args()

    print("inputfile:", args.input)
    print("outputfile:", args.output)

    schroedinger_dir = "/data/shared/software/schrodinger2021-1/"
    convert = f"{schroedinger_dir}/utilities/sdconvert"

    # check that file is present
    if not os.path.isfile(f"{args.input}"):
        raise RuntimeError(f"{args.input} file not found")

    # convert to mae file
    # http://gohom.win/ManualHom/Schrodinger/Schrodinger_2015-2_docs/ligprep/ligprep_user_manual.pdf
    o = subprocess.run(
        [convert, "-imae", args.input, "-osdf", args.output, "-annstereo", "-pKa"],
        stderr=subprocess.STDOUT,
    )
    o.check_returncode()


if __name__ == "__main__":
    main()
