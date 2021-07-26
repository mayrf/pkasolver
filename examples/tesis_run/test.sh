#$ -S /bin/bash
#$ -M e1425153@student.tuwien.ac.at
#$ -m e
#$ -j y
#$ -p -700
#$ -pe smp 1
#$ -o /data/shared/projects/SGE_LOG/
#$ -l gpu=1

# anaconda env definition
. /data/shared/projects/pKa-prediction/anaconda3/etc/profile.d/conda.sh
conda activate py38 
#######################

NUMBER=$4
MODE=$2
EDGE=$3
MODEL=$1

# anaconda env definition
. /data/shared/projects/pKa-prediction/anaconda3/etc/profile.d/conda.sh
conda activate py38
########################

cd /home/fmayr/Documents/Diplomarbeit_Mayr/pkasolver/examples/run15_relu_feat                                               

python train_script.py $MODEL $MODE $EDGE $NUMBER
