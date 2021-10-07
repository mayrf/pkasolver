from pkasolver.ml_architecture import (
    GCNPairSingleConv,
    GCNPairTwoConv,
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
        "pair": {"no-edge": [GCNPairSingleConv, GCNPairTwoConv], "edge": NNConvPair,},
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
    model1, model2 = gcn_dict["pair"]["no-edge"]
    print(model1)
    model1(num_node_features=6, num_edge_features=2)
    print(model2)
    model1(num_node_features=6, num_edge_features=2)


def test_train_gcn_models():
    from pkasolver.ml_architecture import gcn_train, gcn_test
    from pkasolver.data import (
        load_data,
        preprocess,
        make_pyg_dataset_from_dataframe,
    )
    from pkasolver.ml import dataset_to_dataloader

    gcn_dict = {
        "prot": {"no-edge": [GCNProt], "edge": [NNConvProt]},
        "deprot": {"no-edge": [GCNDeprot], "edge": [NNConvDeprot]},
        "pair": {"no-edge": [GCNPairSingleConv, GCNPairTwoConv], "edge": [NNConvPair],},
    }

    print(gcn_dict.values())

    sdf_filepaths = load_data()
    df = preprocess(sdf_filepaths["Novartis"])

    # number of node/edge features
    list_n = ["atomic_number", "formal_charge"]
    list_e = ["bond_type", "is_conjugated"]
    # start with generating datasets based on charge

    # generated PairedData set
    dataset = make_pyg_dataset_from_dataframe(df, list_n, list_e, paired=True)
    dataloader = dataset_to_dataloader(dataset, batch_size=64, shuffle=False)

    for model_raws in [
        gcn_dict["deprot"]["edge"],
        gcn_dict["prot"]["edge"],
        gcn_dict["deprot"]["no-edge"],
        gcn_dict["prot"]["no-edge"],
        gcn_dict["pair"]["no-edge"],
        gcn_dict["pair"]["edge"],
    ]:
        #################
        # test single models
        for model_raw in model_raws:
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

                gcn_train(model, dataloader, optimizer)
                gcn_test(model, dataloader)
                gcn_test(model, dataloader)

    #################################
    # Repeat with different number of edge/nodde features
    #################################
    # number of node/edge features
    list_n = ["atomic_number", "formal_charge", "chiral_tag", "hybridization"]
    list_e = ["bond_type", "is_conjugated"]
    # start with generating datasets based on charge

    # generated PairedData set
    dataset = make_pyg_dataset_from_dataframe(df, list_n, list_e, paired=True)
    dataloader = dataset_to_dataloader(dataset, batch_size=64, shuffle=False)

    for model_raws in [
        gcn_dict["deprot"]["edge"],
        gcn_dict["prot"]["edge"],
        gcn_dict["deprot"]["no-edge"],
        gcn_dict["prot"]["no-edge"],
        gcn_dict["pair"]["no-edge"],
        gcn_dict["pair"]["edge"],
    ]:
        #################
        # test single models
        for model_raw in model_raws:
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

                gcn_train(model, dataloader, optimizer)
                gcn_test(model, dataloader)
                gcn_test(model, dataloader)
