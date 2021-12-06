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

# cd ${path}
# pwd
# hostname

version=0
# data_path='/data/shared/projects/pkasolver-data'
data_path='/data/shared/projects/pkasolver-data-clean'

python 07_test_set_performance.py --testset ${data_path}/05_novartis_testdata.pkl --name GINPairV1_hp_novartis --model ${data_path}/training_with_GINPairV1_v*_hp 
python 07_test_set_performance.py --testset ${data_path}/05_AvLiLuMoVe_testdata.pkl --name GINPairV1_hp_AvLiLuMoVe --model ${data_path}/training_with_GINPairV1_v*_hp

python 07_test_set_performance.py --testset ${data_path}/05_novartis_testdata.pkl --name GINPairV1_lp_novartis --model ${data_path}/training_with_GINPairV1_v*_lp 
python 07_test_set_performance.py --testset ${data_path}/05_AvLiLuMoVe_testdata.pkl --name GINPairV1_lp_AvLiLuMoVe --model ${data_path}/training_with_GINPairV1_v*_lp 

python 07_test_set_performance.py --testset ${data_path}/05_novartis_testdata.pkl --name GINPairV2_hp_novartis --model ${data_path}/training_with_GINPairV2_v*_hp 
python 07_test_set_performance.py --testset ${data_path}/05_AvLiLuMoVe_testdata.pkl --name GINPairV2_hp_AvLiLuMoVe --model ${data_path}/training_with_GINPairV2_v*_hp 

python 07_test_set_performance.py --testset ${data_path}/05_novartis_testdata.pkl --name GINPairV2_lp_novartis --model ${data_path}/training_with_GINPairV2_v*_lp 
python 07_test_set_performance.py --testset ${data_path}/05_AvLiLuMoVe_testdata.pkl --name GINPairV2_lp_AvLiLuMoVe --model ${data_path}/training_with_GINPairV2_v*_lp 