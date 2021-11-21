#! /usr/bin/bash

version=0
data_path='/data/shared/projects/pkasolver-data/'

python 00_downdload_mols_from_chembl.py --output /data/local/00_mols_chembl_v${version}.sdf.gz
python 01_convert_sdf_to_mae.py --input /data/local/00_mols_chembl_v${version}.sdf.gz --output {data_path}/01_mols_chembl_v${version}.mae.gz
python 02_predict_pka_with_epik.py --input {data_path}/01_mols_chembl_v${version}.mae.gz --output {data_path}/02_mols_chembl_with_pka_v${version}.mae.gz
python 03_convert_mae_to_sdf.py --input {data_path}/02_mols_chembl_with_pka_v${version}.mae.gz --output {data_path}/03_mols_chembl_with_pka_v${version}.sdf.gz
python 04_split_epik_output.py --input {data_path}/03_mols_chembl_with_pka_v${version}.sdf.gz --output {data_path}/04_split_mols_chembl_with_pka_v${version}.sdf
python 05_data_preprocess.py --input ${data_path}/04_split_mols_v1.sdf --output ${data_path}/05_chembl_pretrain_data_v${version}.pkl
# the following calls need some rework
# NOTE: we have this in the repository and should use the files from there
python 05_data_preprocess.py --input ${data_path}/04_novartis_cleaned_mono_unique_notraindata.sdf --output ${data_path}/05_novartis_cleaned_mono_unique_notraindata.pkl 
python 05_data_preprocess.py --input ${data_path}/04_AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf --output ${data_path}/05_AvLiLuMoVe_cleaned_mono_unique_notraindata.pkl
python 05_data_preprocess.py --input ${data_path}/04_combined_training_datasets_unique.sdf --output ${data_path}/05_combined_training_datasets_unique.pkl
python 06_training.py --input /data/shared/projects/pkasolver-data/05_chembl_pretrain_data_v1.pkl --model /data/shared/projects/pkasolver-data/06_pretrained_model.pkl
python 06_training.py --input /data/shared/projects/pkasolver-data/05_combined_training_datasets_unique.pkl --model /data/shared/projects/pkasolver-data/06_model_fully_trained.pkl -r
python 07_test_set_performance.py --input /data/shared/projects/pkasolver-data/05_novartis_cleaned_mono_unique_notraindata.pkl --model /data/shared/projects/pkasolver-data/06_model_fully_trained.pkl -r
python 07_test_set_performance.py --input /data/shared/projects/pkasolver-data/05_AvLiLuMoVe_cleaned_mono_unique_notraindata.pkl.pkl --model /data/shared/projects/pkasolver-data/06_model_fully_trained.pkl -r
