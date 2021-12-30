#$ -S /bin/bash
# #$ -M marcus.wieder@univie.ac.at
# #$ -m e
# #$ -j y
# #$ -p -500
# #$ -o /data/shared/projects/SGE_LOG/
# #$ -l gpu=1

# . /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
# conda activate pkasolver

path=$1

data_path='/data/shared/projects/pkasolver-data-clean'

python 07_test_set_performance.py --testset ${data_path}/05_novartis_testdata_pyg_data.pkl --name GINPairV1_hp_novartis --model ${data_path}/trained_models_v1/training_with_GINPairV1_v*_hp/reg_everything_best_model.pt
python 07_test_set_performance.py --testset ${data_path}/05_AvLiLuMoVe_testdata_pyg_data.pkl --name GINPairV1_hp_AvLiLuMoVe --model ${data_path}/trained_models_v1/training_with_GINPairV1_v*_hp/reg_everything_best_model.pt