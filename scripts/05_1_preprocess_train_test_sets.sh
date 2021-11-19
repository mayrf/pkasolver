#! /usr/bin/bash

version=1
data_path=/data/shared/projects/pkasolver-data


python 05_data_preprocess.py --input ${data_path}/04_novartis_cleaned_mono_unique_notraindata.sdf --output ${data_path}/05_novartis_cleaned_mono_unique_notraindata.pkl 
python 05_data_preprocess.py --input ${data_path}/04_AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf --output ${data_path}/05_AvLiLuMoVe_cleaned_mono_unique_notraindata.pkl
python 05_data_preprocess.py --input ${data_path}/04_combined_training_datasets_unique.sdf --output ${data_path}/05_combined_training_datasets_unique.pkl