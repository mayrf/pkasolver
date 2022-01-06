#!/bin/sh
# define conda path
. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
# acivate conda environment
conda activate pkasolver


path=$1

# path must be changed to working directory
data_path='/data/shared/projects/pkasolver-data-clean-pickled-models'

python 07_test_set_performance.py --testset ${data_path}/05_novartis_testdata_pyg_data.pkl --name GINPairV1_hp_novartis --model ${data_path}/trained_models_v1/training_with_GINPairV1_v*_hp/reg_everything_best_model.pkl
python 07_test_set_performance.py --testset ${data_path}/05_AvLiLuMoVe_testdata_pyg_data.pkl --name GINPairV1_hp_AvLiLuMoVe --model ${data_path}/trained_models_v1/training_with_GINPairV1_v*_hp/reg_everything_best_model.pkl