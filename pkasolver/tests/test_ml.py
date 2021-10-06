from pkasolver.ml_architecture import (
    GCNPair,
    GCNProt,
    GCNDeprot,
    NNConvPair,
    NNConvDeprot,
    NNConvProt,
    gcn_full_training,
)
import torch
from pkasolver.constants import DEVICE


def test_init_gcn_models():

    gcn_dict = {
        "prot": {"no-edge": GCNProt, "edge": NNConvProt},
        "deprot": {"no-edge": GCNDeprot, "edge": NNConvDeprot},
        "pair": {"no-edge": GCNPair, "edge": NNConvPair},
    }

    #################
    # test single models
    model = gcn_dict["prot"]["edge"]
    print(model)
    model(num_node_features=6, num_edge_features=2)
    model = gcn_dict["prot"]["no-edge"]
    print(model)
    model(num_node_features=6, num_edge_features=2)
    model = gcn_dict["deprot"]["edge"]
    print(model)
    model(num_node_features=6, num_edge_features=2)
    model = gcn_dict["deprot"]["no-edge"]
    print(model)
    model(num_node_features=6, num_edge_features=2)
    #################
    #  test pair models
    model = gcn_dict["pair"]["edge"]
    print(model)
    model(num_node_features=6, num_edge_features=2)
    model = gcn_dict["pair"]["no-edge"]
    print(model)
    model(num_node_features=6, num_edge_features=2)


def test_train_gcn_models():

    gcn_dict = {
        "prot": {"no-edge": GCNProt, "edge": NNConvProt},
        "deprot": {"no-edge": GCNDeprot, "edge": NNConvDeprot},
        "pair": {"no-edge": GCNPair, "edge": NNConvPair},
    }

    print(gcn_dict.values())
    from pkasolver.ml_architecture import gcn_train
    from pkasolver.data import (
        load_data,
        preprocess,
        make_pyg_dataset_from_dataframe,
    )
    from pkasolver.ml import dataset_to_dataloader

    sdf_filepaths = load_data()
    df = preprocess(sdf_filepaths["Novartis"])
    list_n = ["atomic_number", "formal_charge"]
    list_e = ["bond_type", "is_conjugated"]
    # start with generating datasets based on charge

    # generated PairedData set
    dataset = make_pyg_dataset_from_dataframe(df, list_n, list_e, paired=True)
    l = dataset_to_dataloader(dataset, batch_size=64, shuffle=False)

    for model_raw in [
        gcn_dict["deprot"]["edge"],
        gcn_dict["prot"]["edge"],
        gcn_dict["deprot"]["no-edge"],
        gcn_dict["prot"]["no-edge"],
        gcn_dict["pair"]["no-edge"],
        gcn_dict["pair"]["edge"],
    ]:
        #################
        # test single models
        print(model_raw)

        for attention_mode in [False, True]:
            print(attention_mode)
            model = model_raw(
                num_node_features=len(list_n),
                num_edge_features=len(list_e),
                attention=attention_mode,
            ).to(device=DEVICE)
            print(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

            gcn_train(model, l, optimizer)
