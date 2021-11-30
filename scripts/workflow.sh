#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -500
#$ -o /data/shared/projects/SGE_LOG/
#$ -l gpu=1

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate pkasolver

path=$1

cd ${path}
pwd
hostname

version=0
data_path='/data/shared/projects/pkasolver-data-clean/'
#python 00_downdload_mols_from_chembl.py --output /data/local/00_mols_chembl_v${version}.sdf.gz
#python 01_convert_sdf_to_mae.py --input /data/local/00_mols_chembl_v${version}.sdf.gz --output ${data_path}/01_mols_chembl_v${version}.mae.gz
#python 02_predict_pka_with_epik.py --input ${data_path}/01_mols_chembl_v${version}.mae.gz --output ${data_path}/02_mols_chembl_with_pka_v${version}.mae.gz
#python 03_convert_mae_to_sdf.py --input ${data_path}/02_mols_chembl_with_pka_v${version}.mae.gz --output ${data_path}/03_mols_chembl_with_pka_v${version}.sdf.gz
#python 04_0_filter_testmols.py --input ${data_path}/03_mols_chembl_with_pka_v${version}.sdf.gz  --output ${data_path}/04_mols_chembl_with_pka_v${version}_filtered.sdf.gz --filter ${data_path}/00_AvLiLuMoVe_testdata.sdf,${data_path}/00_novartis_testdata.sdf
#python 04_1_split_epik_output.py --input ${data_path}/04_mols_chembl_with_pka_v${version}_filtered.sdf.gz --output ${data_path}/04_split_mols_chembl_with_pka_v${version}.sdf.gz
#python 05_data_preprocess.py --input ${data_path}/04_split_mols_chembl_with_pka_v${version}.sdf.gz --output ${data_path}/05_chembl_pretrain_data_v${version}.pkl
#python 05_data_preprocess.py --input ${data_path}/00_novartis_testdata.sdf --output ${data_path}/05_novartis_testdata.pkl 
#python 05_data_preprocess.py --input ${data_path}/00_AvLiLuMoVe_testdata.sdf --output ${data_path}/05_AvLiLuMoVe_testdata.pkl
#python 05_data_preprocess.py --input ${data_path}/00_experimental_training_datasets.sdf --output ${data_path}/05_experimental_training_datasets.pkl
# start with pretraining on the CHEMBL data
# you have to specify the GNN model in the script 
#python 06_training.py --input ${data_path}/05_chembl_pretrain_data_v${version}.pkl --model ${data_path}/06_pretrained_model_v${version}.pkl --epoch 1000
# transfer learning on the experimental data
#you have to import the specific GNN model in the script
#python 06_training.py --input ${data_path}/05_experimental_training_datasets.pkl --model ${data_path}/06_pretrained_model_v${version}_GINPairV2.pkl -r --epoch 300
# evaluate performance on the test set
python 07_test_set_performance.py --testset ${data_path}/05_novartis_testdata.pkl --model ${data_path}/06_pretrained_model_v${version}_GINPairV2_retrained.pkl
python 07_test_set_performance.py --testset ${data_path}/05_AvLiLuMoVe_testdata.pkl --model ${data_path}/06_pretrained_model_v${version}_GINPairV2_retrained.pkl
