# define conda path
. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
# acivate conda environment
conda activate pkasolver

# ??
pwd
hostname
# ??

version=0
echo ${version}

# path must be changed to working directory
data_path='/data/shared/projects/pkasolver-data-test'

# downloading mols from the ChEMBL
python 00_download_mols_from_chembl.py --output ${data_path}/00_mols_chembl_v${version}.sdf.gz
# convert to mae input format
python 01_convert_sdf_to_mae.py --input ${data_path}/00_mols_chembl_v${version}.sdf.gz --output ${data_path}/01_mols_chembl_v${version}.mae.gz
# predict microstate pKa values with EPIK
python 02_predict_pka_with_epik.py --input ${data_path}/01_mols_chembl_v${version}.mae.gz --output ${data_path}/02_mols_chembl_with_pka_v${version}.mae.gz
# convert to sdf file format
python 03_convert_mae_to_sdf.py --input ${data_path}/02_mols_chembl_with_pka_v${version}.mae.gz --output ${data_path}/03_mols_chembl_with_pka_v${version}.sdf.gz
filter mols that are present in test sets
python 04_0_filter_testmols.py --input ${data_path}/03_mols_chembl_with_pka_v${version}.sdf.gz  --output ${data_path}/04_mols_chembl_with_pka_v${version}_filtered.sdf.gz --filter ${data_path}/00_AvLiLuMoVe_testdata.sdf,${data_path}/00_novartis_testdata.sdf
# split mols with muliple pkas  
python 04_1_split_epik_output.py --input ${data_path}/04_mols_chembl_with_pka_v${version}_filtered.sdf.gz --output ${data_path}/04_split_mols_chembl_with_pka_v${version}_mols.pkl
# generate pyg data
python 05_data_preprocess.py --input ${data_path}/04_split_mols_chembl_with_pka_v${version}_mols.pkl --output ${data_path}/05_chembl_pretrain_data_v${version}_pyg_data.pkl