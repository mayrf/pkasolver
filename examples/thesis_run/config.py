#General
SEED=42
#raw data paths

raw_data_dir ="../../data/Baltruschat/"
SDF_TRAIN= raw_data_dir + "combined_training_datasets_unique.sdf"
TRAIN_NAME="Baltruschat"

SDF_TEST=[raw_data_dir + "novartis_cleaned_mono_unique_notraindata.sdf", raw_data_dir + "AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf"]
TEST_NAMES=["Novartis","Literature"]

#PAIRED = False
TRAIN_TEST_SPLIT = 0.8

#Morgan fingerprints
FP_BITS=4096
FP_RADIUS=3

#Random Forest
NUM_ESTIMATORS=100 #1000 Baltruschat

#GCN
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 2000 # 3000
DEVICE="cpu"
NUM_GRAPH_LAYERS = 4
NUM_LINEAR_LAYERS = 2
HIDDEN_CHANNELS = 96

NODE_FEATURES = [
    'atomic_number',
    'formal_charge',
    'chiral_tag',
    'hybridization',
    'total_num_Hs',
    'explicit_num_Hs',
    'aromatic_tag',
    'total_valence',
    'total_degree',
    'is_in_ring',
    'amide_center_atom'
]
EDGE_FEATURES = [
    'bond_type',
    'is_conjugated',
    'rotatable'
]

num_node_features = len(NODE_FEATURES)
num_edge_features = len(EDGE_FEATURES)
best_loss=1

