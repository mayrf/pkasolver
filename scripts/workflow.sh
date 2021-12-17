#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -500
#$ -o /data/shared/projects/SGE_LOG/
#$ -l gpu=1
#$ -cwd
#$ -l h='!(node01|node12)'

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate pkasolver
parameter_size='hp'
model_name='GINPairV1'
pwd
hostname

version=$1
echo ${version}
data_path='/data/shared/projects/pkasolver-data-clean_final/'
python 00_downdload_mols_from_chembl.py --output /data/local/00_mols_chembl_v${version}.sdf.gz
python 01_convert_sdf_to_mae.py --input /data/local/00_mols_chembl_v${version}.sdf.gz --output ${data_path}/01_mols_chembl_v${version}.mae.gz
python 02_predict_pka_with_epik.py --input ${data_path}/01_mols_chembl_v${version}.mae.gz --output ${data_path}/02_mols_chembl_with_pka_v${version}.mae.gz
python 03_convert_mae_to_sdf.py --input ${data_path}/02_mols_chembl_with_pka_v${version}.mae.gz --output ${data_path}/03_mols_chembl_with_pka_v${version}.sdf.gz
python 04_0_filter_testmols.py --input ${data_path}/03_mols_chembl_with_pka_v${version}.sdf.gz  --output ${data_path}/04_mols_chembl_with_pka_v${version}_filtered.sdf.gz --filter ${data_path}/00_AvLiLuMoVe_testdata.sdf,${data_path}/00_novartis_testdata.sdf
python 04_1_split_epik_output.py --input ${data_path}/04_mols_chembl_with_pka_v${version}_filtered.sdf.gz --output ${data_path}/04_split_mols_chembl_with_pka_v${version}_mols.pkl
python 05_data_preprocess.py --input ${data_path}/04_split_mols_chembl_with_pka_v${version}_mols.pkl --output ${data_path}/05_chembl_pretrain_data_vv${version}_pyg_data.pkl
python 04_2_prepare_rest.py --input ${data_path}/00_novartis_testdata.sdf --output ${data_path}/04_novartis_testdata_mols.pkl
python 05_data_preprocess.py --input ${data_path}/04_novartis_testdata_mols.pkl --output ${data_path}/05_novartis_testdata_pyg_data.pkl 
python 04_2_prepare_rest.py --input ${data_path}/00_AvLiLuMoVe_testdata.sdf --output ${data_path}/04_AvLiLuMoVe_testdata_mols.pkl
python 05_data_preprocess.py --input ${data_path}/04_AvLiLuMoVe_testdata_mols.pkl --output ${data_path}/05_AvLiLuMoVe_testdata_pyg_data.pkl 
python 04_2_prepare_rest.py --input ${data_path}/00_experimental_training_datasets.sdf --output ${data_path}/04_experimental_training_datasets_mols.pkl
python 05_data_preprocess.py --input ${data_path}/04_experimental_training_datasets_mols.pkl --output ${data_path}/05_experimental_training_datasets_pyg_data.pkl 
python 05_data_preprocess.py --input ${data_path}/00_AvLiLuMoVe_testdata.sdf --output ${data_path}/05_AvLiLuMoVe_testdata.pkl
python 05_data_preprocess.py --input ${data_path}/00_experimental_training_datasets.sdf --output ${data_path}/05_experimental_training_datasets.pkl
#start with pretraining on the CHEMBL data
python 06_training.py --input ${data_path}/05_chembl_pretrain_data_v0.pkl --model ${data_path}/trained_models_v0/training_with_${model_name}_v${version}_${parameter_size} --model_name ${model_name} --epoch 1000
# transfer learning on the experimental data
python 06_training.py --input ${data_path}/05_experimental_training_datasets.pkl --model ${data_path}/trained_models_v1/training_with_${model_name}_v${version}_${parameter_size} -r --epoch 1_000 --model_name ${model_name}
#python 06_training.py --input ${data_path}/05_experimental_training_datasets.pkl --model ${data_path}/ONLY_fined_tuned_models_v0/training_with_${model_name}_v${version}_${parameter_size} --epoch 1_000 --model_name ${model_name}
