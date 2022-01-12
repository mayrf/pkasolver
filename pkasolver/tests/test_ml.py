import torch
from pkasolver.constants import DEVICE, edge_feat_values
from pkasolver.data import calculate_nr_of_features
from pkasolver.ml_architecture import (GATPair, GATProt, GCNDeprot,
                                       GCNPairSingleConv, GCNPairTwoConv,
                                       GCNProt, GINPairV1, GINPairV2,
                                       GINPairV3, GINProt, NNConvDeprot,
                                       NNConvPair, NNConvProt)

models = [
    ("GCNPairSingleConv", GCNPairSingleConv),
    ("GCNPairTwoConv", GCNPairTwoConv),
    ("GCNProt", GCNProt),
    ("GCNDeprot", GCNDeprot),
    ("NNConvPair", NNConvPair),
    ("NNConvDeprot", NNConvDeprot),
    ("NNConvProt", NNConvProt),
    # ("GINProt", GINProt),
]


def test_init_gcn_models():

    #################
    # test single models
    for model_name, model_class in models:
        print(model_name)
        model = model_class
        print(model)
        model(num_node_features=6, num_edge_features=2)


def test_train_gcn_models():
    from pkasolver.data import (load_data, make_pyg_dataset_from_dataframe,
                                preprocess)
    from pkasolver.ml import dataset_to_dataloader
    from pkasolver.ml_architecture import gcn_test, gcn_train

    sdf_filepaths = load_data()
    df = preprocess(sdf_filepaths["Novartis"])

    # number of node/edge features
    list_n = ["element", "formal_charge"]
    list_e = ["bond_type", "is_conjugated"]
    # start with generating datasets based on charge

    # generated PairedData set
    dataset = make_pyg_dataset_from_dataframe(df, list_n, list_e, paired=True)
    dataloader = dataset_to_dataloader(dataset, batch_size=64, shuffle=False)

    # calculate node/edge features
    num_node_features = calculate_nr_of_features(list_n)
    num_edge_features = calculate_nr_of_features(list_e)

    for model_name, model_class in models:
        print(model_name)
        print(model_class)  #################
        # test single models

        for attention_mode in [False, True]:
            print(attention_mode)
            model = model_class(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                attention=attention_mode,
            ).to(device=DEVICE)
            print(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

            gcn_train(model, dataloader, optimizer)
            gcn_test(model, dataloader)

    #################################
    # Repeat with different number of edge/nodde features
    #################################
    # number of node/edge features
    list_n = ["element", "formal_charge", "aromatic_tag", "hybridization"]
    list_e = ["bond_type", "is_conjugated"]
    # start with generating datasets based on charge

    # generated PairedData set
    dataset = make_pyg_dataset_from_dataframe(df, list_n, list_e, paired=True)
    dataloader = dataset_to_dataloader(dataset, batch_size=64, shuffle=False)

    # calculate node/edge features
    num_node_features = calculate_nr_of_features(list_n)
    num_edge_features = calculate_nr_of_features(list_e)

    i_e = 0
    for feat in list_e:
        i_e += len(edge_feat_values[feat])
    num_edge_features = i_e

    for model_name, model_class in models:
        print(model_name)
        print(model_class)  #################
        # test single models
        #################
        # test single models

        for attention_mode in [False, True]:
            print(attention_mode)
            model = model_class(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                hidden_channels=64,
                attention=attention_mode,
            ).to(device=DEVICE)
            print(model)
            print(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

            gcn_train(model, dataloader, optimizer)
            gcn_test(model, dataloader)
            gcn_test(model, dataloader)


new_models = [
    ("GINProt", GINProt),
    ("GINPairV1", GINPairV1),
    ("GINPairV2", GINPairV2),
    ("GATProt", GATProt),
    ("GATPair", GATPair),
]


def test_train_new_models():
    from pkasolver.data import (load_data, make_pyg_dataset_from_dataframe,
                                preprocess)
    from pkasolver.ml import dataset_to_dataloader
    from pkasolver.ml_architecture import gcn_test, gcn_train

    sdf_filepaths = load_data()
    df = preprocess(sdf_filepaths["Novartis"])

    # number of node/edge features
    list_n = [
        "element",
        "formal_charge",
        "hybridization",
        "total_num_Hs",
        "aromatic_tag",
        "total_valence",
        "total_degree",
        "is_in_ring",
        "reaction_center",
    ]
    list_e = [
        "bond_type",
        "is_conjugated",
        "rotatable",
    ]  # start with generating datasets based on charge

    # generated PairedData set
    dataset = make_pyg_dataset_from_dataframe(df, list_n, list_e, paired=True)
    dataloader = dataset_to_dataloader(dataset, batch_size=64, shuffle=False)
    # calculate node/edge features
    num_node_features = calculate_nr_of_features(list_n)
    num_edge_features = calculate_nr_of_features(list_e)

    for model_name, model_class in new_models:
        print(model_name)
        print(model_class)  #################
        # test single models

        for attention_mode in [False, True]:
            print(attention_mode)
            model = model_class(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                hidden_channels=64,
                attention=attention_mode,
            ).to(device=DEVICE)
            print(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

            gcn_train(model, dataloader, optimizer)
            gcn_test(model, dataloader)


new_models = [
    ("GINPairV1", GINPairV1),
    ("GINPairV3", GINPairV3),
]


def test_only_GINPairV1_and_GINPairV3_models():
    from pkasolver.data import (load_data, make_pyg_dataset_from_dataframe,
                                preprocess)
    from pkasolver.ml import dataset_to_dataloader
    from pkasolver.ml_architecture import gcn_test, gcn_train

    sdf_filepaths = load_data()
    df = preprocess(sdf_filepaths["Novartis"])

    # number of node/edge features
    list_n = [
        "element",
        "formal_charge",
        "hybridization",
        "total_num_Hs",
        "aromatic_tag",
        "total_valence",
        "total_degree",
        "is_in_ring",
        "reaction_center",
        "smarts",
    ]
    list_e = [
        "bond_type",
        "is_conjugated",
        "rotatable",
    ]  # start with generating datasets based on charge

    # generated PairedData set
    dataset = make_pyg_dataset_from_dataframe(df, list_n, list_e, paired=True)
    dataloader = dataset_to_dataloader(dataset, batch_size=64, shuffle=False)
    # calculate node/edge features
    num_node_features = calculate_nr_of_features(list_n)
    num_edge_features = calculate_nr_of_features(list_e)

    for model_name, model_class in new_models:
        print(model_name)
        print(model_class)
        #################
        # test single models

        model = model_class(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_channels=96,
            attention=False,
        ).to(device=DEVICE)
        print(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        gcn_train(model, dataloader, optimizer)
        gcn_test(model, dataloader)
        print("####################")
        print(list(model.named_modules()))
        print("####################")
        print(model.get_submodule("GIN_d"))
        print("####################")
        print(model.get_submodule("lins.2"))
        print("####################")
        print(model.get_submodule("final_lin"))
        print("####################")

        print(
            "Number of parameters: ",
            sum(p.numel() for p in model.parameters() if p.requires_grad == True),
        )


def test_only_GINProt_models():
    new_models = [
        ("GINProt", GINProt),
    ]

    from pkasolver.data import (load_data, make_pyg_dataset_from_dataframe,
                                preprocess)
    from pkasolver.ml import dataset_to_dataloader
    from pkasolver.ml_architecture import gcn_test, gcn_train

    sdf_filepaths = load_data()
    df = preprocess(sdf_filepaths["Novartis"])

    # number of node/edge features
    list_n = [
        "element",
        "formal_charge",
        "hybridization",
        "total_num_Hs",
        "aromatic_tag",
        "total_valence",
        "total_degree",
        "is_in_ring",
        "reaction_center",
        "smarts",
    ]
    list_e = [
        "bond_type",
        "is_conjugated",
        "rotatable",
    ]  # start with generating datasets based on charge

    # generated PairedData set
    dataset = make_pyg_dataset_from_dataframe(df, list_n, list_e, paired=True)
    dataloader = dataset_to_dataloader(dataset, batch_size=64, shuffle=False)
    # calculate node/edge features
    num_node_features = calculate_nr_of_features(list_n)
    num_edge_features = calculate_nr_of_features(list_e)

    for model_name, model_class in new_models:
        print(model_name)
        print(model_class)
        #################
        # test single models

        model = model_class(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_channels=128,
            out_channels=64,
            attention=False,
        ).to(device=DEVICE)
        print(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        gcn_train(model, dataloader, optimizer)
        gcn_test(model, dataloader)
        print("####################")
        print(list(model.named_modules()))
        print("####################")
        print(model.get_submodule("lins.2"))
        print("####################")
        print(model.get_submodule("final_lin"))
        print("####################")

        print(
            "Number of parameters: ",
            sum(p.numel() for p in model.parameters() if p.requires_grad == True),
        )


def test_count_nr_of_parameters():

    import torch.nn.functional as F
    from torch.nn import Linear, ModuleList

    lins = []
    lins.append(Linear(64, 96))
    for _ in range(0, 2):
        lins.append(Linear(96, 96))
    lins.append(Linear(96, 1))
    model = ModuleList(lins)
    print(model)
    nr_of_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad == True
    )
    print(f"Number of parameters: {nr_of_parameters=}")


def test_count_nr_of_parameters_for_GIN():

    import torch.nn.functional as F
    from torch_geometric.nn.models import GAT, GIN, AttentiveFP

    list_n = [
        "element",
        "formal_charge",
        "hybridization",
        "total_num_Hs",
        "aromatic_tag",
        "total_valence",
        "total_degree",
        "is_in_ring",
        "reaction_center",
        "smarts",
    ]
    list_e = [
        "bond_type",
        "is_conjugated",
        "rotatable",
    ]  # start with generating datasets based on charge

    # calculate node/edge features
    num_node_features = calculate_nr_of_features(list_n)

    model = GIN(
        in_channels=num_node_features,
        out_channels=32,
        hidden_channels=96,
        num_layers=4,
        dropout=0.5,
    )

    nr_of_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad == True
    )
    print(f"Number of parameters: {nr_of_parameters=}")
