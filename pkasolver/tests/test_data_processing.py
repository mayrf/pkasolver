import pkasolver
import pytest
import sys


def load_data():
    from ..data import preprocess_all

    base = "data/baltruschat"
    sdf_training = f"{base}/combined_training_datasets_unique.sdf"
    sdf_novartis = f"{base}/novartis_cleaned_mono_unique_notraindata.sdf"
    sdf_AvLiLuMoVe = f"{base}/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf"

    datasets = {
        "Training": sdf_training,
        "Novartis": sdf_novartis,
        "AvLiLuMoVe": sdf_AvLiLuMoVe,
    }
    return preprocess_all(datasets)


def test_load_data():
    load_data()


def test_make_feature_list():
    from ..data import make_features_dicts
    from ..constants import NODE_FEATURES, EDGE_FEATURES

    node_features = [
        "atomic_number",
        "hybridization",
    ]
    edge_features = ["bond_type", "is_conjugated"]

    f = make_features_dicts(NODE_FEATURES, node_features)
    assert "atomic_number" in f.keys()
    assert "hybridization" in f.keys()
    f = make_features_dicts(EDGE_FEATURES, edge_features)
    assert "bond_type" in f.keys()
    assert "is_conjugated" in f.keys()


def generate_pairwise_data():
    from ..data import make_pyg_dataset

    node_features = [
        "atomic_number",
        "hybridization",
    ]
    edge_features = ["bond_type", "is_conjugated"]
    dataset = load_data()
    df = dataset["Training"]
    return make_pyg_dataset(df, node_features, edge_features)


def test_generate_pairwise_data():
    dataset = generate_pairwise_data()
    print(dataset)
