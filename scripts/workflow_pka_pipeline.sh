#! /usr/bin/bash

version=1
data_path=/data/shared/projects/pkasolver-data

# python 05_data_preprocess.py --input ${data_path}/04_split_mols_v${version}.sdf --output ${data_path}/05_chembl_pretrain_test_data_v${version}.pkl
python 06_pretraining.py --input ${data_path}/05_chembl_pretrain_data_v${version}_subset.pkl --val ${data_path}/05_combined_training_datasets_unique.pkl --output ${data_path}/06_model_pretrained_w_chembl_v${version}.pkl