#!/bin/sh
# define conda path
. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
# acivate conda environment
conda activate pkasolver


parameter_size='hp'
model_name='GINPairV1'

# ??
pwd
hostname
# ??

version=$1
echo ${version}

# path must be changed to working directory
data_path='/data/shared/projects/pkasolver-data-test'

# start with pretraining on the CHEMBL data
# python 06_training.py --input ${data_path}/05_chembl_pretrain_data_v0_pyg_data.pkl --model ${data_path}/trained_models_v0/training_with_${model_name}_v${version}_${parameter_size} --model_name ${model_name} --epoch 10
# transfer learning on the experimental data
python 06_training.py --input ${data_path}/05_experimental_training_datasets_pyg_data.pkl --model ${data_path}/trained_models_v0/training_with_${model_name}_v${version}_${parameter_size} -r --epoch 1000 --model_name ${model_name}
#python 06_training.py --input ${data_path}/05_experimental_training_datasets_pyg_data.pkl --model ${data_path}/ONLY_fined_tuned_models_v0/training_with_${model_name}_v${version}_${parameter_size} --epoch 1_000 --model_name ${model_name}
