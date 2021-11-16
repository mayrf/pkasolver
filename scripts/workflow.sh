#! /usr/bin/bash

version=0

python downdload_mols_from_chembl.py --output /data/local/mols_chembl_v${version}.sdf.gz
python convert_sdf_to_mae.py --input /data/local/mols_chembl_v${version}.sdf.gz --output /data/shared/projects/pkasolver-data/mols_chembl_v${version}.mae.gz
python predict_pka_with_epik.py --input /data/shared/projects/pkasolver-data/mols_chembl_v${version}.mae.gz --output /data/shared/projects/pkasolver-data/mols_chembl_with_pka_v${version}.mae.gz
python convert_mae_to_sdf.py --input /data/shared/projects/pkasolver-data/mols_chembl_with_pka_v${version}.mae.gz --output /data/shared/projects/pkasolver-data/mols_chembl_with_pka_v${version}.sdf.gz
python split_epik_output.py --input /data/shared/projects/pkasolver-data/mols_chembl_with_pka_v${version}.sdf.gz --output --output /data/shared/projects/pkasolver-data/split_mols_chembl_with_pka_v${version}.sdf