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
model_name='GINPairV3'
pwd
hostname

version=$1
echo ${version}
data_path='/data/shared/projects/pkasolver-data-test'

python 04_2_prepare_rest.py --input ${data_path}/00_novartis_testdata.sdf --output ${data_path}/04_novartis_testdata_mols.pkl
python 05_data_preprocess.py --input ${data_path}/04_novartis_testdata_mols.pkl --output ${data_path}/05_novartis_testdata_pyg_data.pkl 
python 04_2_prepare_rest.py --input ${data_path}/00_AvLiLuMoVe_testdata.sdf --output ${data_path}/04_AvLiLuMoVe_testdata_mols.pkl
python 05_data_preprocess.py --input ${data_path}/04_AvLiLuMoVe_testdata_mols.pkl --output ${data_path}/05_AvLiLuMoVe_testdata_pyg_data.pkl 
python 04_2_prepare_rest.py --input ${data_path}/00_experimental_training_datasets.sdf --output ${data_path}/04_experimental_training_datasets_mols.pkl
python 05_data_preprocess.py --input ${data_path}/04_experimental_training_datasets_mols.pkl --output ${data_path}/05_experimental_training_datasets_pyg_data.pkl 
