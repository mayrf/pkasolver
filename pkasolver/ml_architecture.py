import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import (GCNConv, GlobalAttention, NNConv,
                                global_mean_pool)
from tqdm import tqdm

from pkasolver.constants import DEVICE, SEED

#####################################
#####################################


def attention_pooling(num_node_features):
    return GlobalAttention(
        Sequential(
            Linear(num_node_features, num_node_features),
            ReLU(),
            Linear(num_node_features, 1),
        )
    )


def forward_convs(x, edge_index, l: list):
    for i in range(len(l)):
        if i < len(l) - 1:
            x = F.relu(l[i](x, edge_index))
        else:
            x = l[i](x, edge_index)
    return x


def forward_convs_with_edge_attr(x, edge_index, edge_attr, l: list):
    for i in range(len(l)):
        if i < len(l) - 1:
            x = F.relu(l[i](x, edge_index, edge_attr))
        else:
            x = l[i](x, edge_index, edge_attr)
    return x


def forward_lins(x, l: list):
    for i in range(len(l)):
        if i < len(l) - 1:
            x = F.relu(l[i](x))
        else:
            x = l[i](x)
    return x


#####################################
#####################################
# defining GCN for single state
#####################################
#####################################
from torch_geometric.nn.models import GAT, GIN, AttentiveFP


class AttentivePka(AttentiveFP):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        dropout: float,
        edge_dim: int,
        num_timesteps: int,
    ):

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
            edge_dim=edge_dim,
        )
        torch.manual_seed(SEED)
        self.checkpoint = {
            "epoch": 0,
            "optimizer_state_dict": "",
            "best_loss": (100, -1, -1),
            "best_states": {},
            "progress_table": {"epoch": [], "train_loss": [], "validation_loss": []},
        }

    @staticmethod
    def _return_lin(
        input_dim: int, nr_of_lin_layers: int, embeding_size: int,
    ):
        lins = []
        lins.append(Linear(input_dim, embeding_size))
        for _ in range(2, nr_of_lin_layers):
            lins.append(Linear(embeding_size, embeding_size))
        lins.append(Linear(embeding_size, 1))
        return ModuleList(lins)


class GATpKa(GAT):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels,
        dropout,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        torch.manual_seed(SEED)
        self.checkpoint = {
            "epoch": 0,
            "optimizer_state_dict": "",
            "best_loss": (100, -1, -1),
            "best_states": {},
            "progress_table": {"epoch": [], "train_loss": [], "validation_loss": []},
        }

    @staticmethod
    def _return_lin(
        input_dim: int, nr_of_lin_layers: int, embeding_size: int,
    ):
        lins = []
        lins.append(Linear(input_dim, embeding_size))
        for _ in range(2, nr_of_lin_layers):
            lins.append(Linear(embeding_size, embeding_size))
        lins.append(Linear(embeding_size, 1))
        return ModuleList(lins)


class GINpKa(GIN):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        dropout: float,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        torch.manual_seed(SEED)

        self.checkpoint = {
            "epoch": 0,
            "optimizer_state_dict": "",
            "best_loss": (100, -1, -1),
            "best_states": {},
            "progress_table": {"epoch": [], "train_loss": [], "validation_loss": []},
        }

    @staticmethod
    def _return_lin(
        input_dim: int, nr_of_lin_layers: int, embeding_size: int,
    ):
        lins = []
        lins.append(Linear(input_dim, embeding_size))
        for _ in range(2, nr_of_lin_layers):
            lins.append(Linear(embeding_size, embeding_size))
        lins.append(Linear(embeding_size, embeding_size))
        return ModuleList(lins)


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        self.checkpoint = {
            "epoch": 0,
            "optimizer_state_dict": "",
            "best_loss": (100, -1, -1),
            "best_states": {},
            "progress_table": {"epoch": [], "train_loss": [], "validation_loss": []},
        }

    @staticmethod
    def _return_lin(
        input_dim: int, nr_of_lin_layers: int, embeding_size: int,
    ):
        lins = []
        lins.append(Linear(input_dim, embeding_size))
        for _ in range(2, nr_of_lin_layers):
            lins.append(Linear(embeding_size, embeding_size))
        lins.append(Linear(embeding_size, 1))
        return ModuleList(lins)

    @staticmethod
    def _return_conv(num_node_features, nr_of_layers, embeding_size):
        convs = []
        convs.append(GCNConv(num_node_features, embeding_size))
        for _ in range(1, nr_of_layers):
            convs.append(GCNConv(embeding_size, embeding_size))
        return ModuleList(convs)

    @staticmethod
    def _return_nnconv(
        num_node_features, num_edge_features, nr_of_layers, embeding_size
    ):

        convs = []
        nn1 = Sequential(
            Linear(num_edge_features, embeding_size),
            ReLU(),
            Linear(embeding_size, num_node_features * embeding_size),
        )
        nn2 = Sequential(
            Linear(num_edge_features, embeding_size),
            ReLU(),
            Linear(embeding_size, embeding_size * embeding_size),
        )
        convs.append(NNConv(num_node_features, embeding_size, nn=nn1))
        for _ in range(1, nr_of_layers):
            convs.append(NNConv(embeding_size, embeding_size, nn=nn2))
        return ModuleList(convs)


#####################################
# tie in classes
# forward function
#####################################
class GCNSingleForward:
    def _forward(self, x, edge_index, x_batch):
        # move batch to device
        x_batch = x_batch.to(device=DEVICE)

        if self.attention:
            # if attention=True, pool
            x_att = self.pool(x, x_batch)

        # run through conv layers
        x = forward_convs(x, edge_index, self.convs)
        # set dimensions to zero
        x = F.dropout(x, p=0.5, training=self.training)

        # global max pooling
        x = global_mean_pool(x, x_batch)  # [batch_size, hidden_channels]

        # if attention=True append attention layer
        if self.attention:
            x = torch.cat((x, x_att), 1)

        # run through linear layer
        x = forward_lins(x, self.lins)
        return x


class GCNPairOneConvForward:
    def _forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        x_p_batch = data.x_p_batch.to(device=DEVICE)
        x_d_batch = data.x_d_batch.to(device=DEVICE)

        # using only a single conv
        x_p = forward_convs(x_p, data.edge_index_p, self.convs)
        x_d = forward_convs(x_d, data.edge_index_d, self.convs)

        x_p = global_mean_pool(x_p, x_p_batch)  # [batch_size, hidden_channels]
        x_d = global_mean_pool(x_d, x_d_batch)

        x_p = F.dropout(x_p, p=0.5, training=self.training)
        x_d = F.dropout(x_d, p=0.5, training=self.training)

        x_p = forward_lins(x_p, self.lins_p)
        x_d = forward_lins(x_d, self.lins_d)

        return x_p + x_d


class GCNPairTwoConvForward:
    def _forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        x_p_batch = data.x_p_batch.to(device=DEVICE)
        x_d_batch = data.x_d_batch.to(device=DEVICE)

        if self.attention:
            x_p_att = self.pool(x_p, x_p_batch)
            x_d_att = self.pool(x_d, x_d_batch)

        x_p = forward_convs(x_p, data.edge_index_p, self.convs_p)
        x_d = forward_convs(x_d, data.edge_index_d, self.convs_d)

        x_p = global_mean_pool(x_p, x_p_batch)  # [batch_size, hidden_channels]
        x_d = global_mean_pool(x_d, x_d_batch)

        if self.attention:
            x = torch.cat((x_p, x_d, x_p_att, x_d_att), 1)
        else:
            x = torch.cat([x_p, x_d], 1)

        x = F.dropout(x, p=0.5, training=self.training)
        x = forward_lins(x, self.lins)
        return x


class NNConvSingleForward:
    def _forward(self, x, x_batch, edge_attr, edge_index):

        x_batch = x_batch.to(device=DEVICE)
        if self.attention:
            x_att = self.pool(x, x_batch)

        x = forward_convs_with_edge_attr(x, edge_index, edge_attr, self.convs)

        x = global_mean_pool(x, x_batch)  # [batch_size, hidden_channels]

        if self.attention:
            x = torch.cat((x, x_att), 1)

        x = F.dropout(x, p=0.5, training=self.training)

        x = forward_lins(x, self.lins)
        return x


class NNConvPairForward:
    def _forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):

        x_p_batch, x_d_batch = (
            data.x_p_batch.to(device=DEVICE),
            data.x_d_batch.to(device=DEVICE),
        )
        x_p_att = self.pool(x_p, x_p_batch)
        x_d_att = self.pool(x_d, x_d_batch)

        x_p = forward_convs_with_edge_attr(
            x_p, data.edge_index_p, edge_attr_p, self.convs_p
        )
        x_d = forward_convs_with_edge_attr(
            x_d, data.edge_index_d, edge_attr_d, self.convs_d
        )

        x_p = global_mean_pool(x_p, x_p_batch)  # [batch_size, hidden_channels]
        x_d = global_mean_pool(x_d, x_d_batch)

        if self.attention:
            x = torch.cat((x_p, x_d, x_p_att, x_d_att), 1)
        else:
            x = torch.cat((x_p, x_d), 1)

        x = F.dropout(x, p=0.5, training=self.training)
        x = forward_lins(x, self.lins)

        return x


class NNConvSingleArchitecture(GCN):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        nr_of_layers: int,
        hidden_channels: int,
    ):
        super().__init__()
        self.pool = attention_pooling(num_node_features)

        self.convs = self._return_nnconv(
            num_node_features,
            num_edge_features,
            nr_of_layers=nr_of_layers,
            embeding_size=hidden_channels,
        )
        if self.attention:
            input_dim = hidden_channels + num_node_features
        else:
            input_dim = hidden_channels

        self.lins = GCN._return_lin(
            input_dim=input_dim, nr_of_lin_layers=2, embeding_size=hidden_channels,
        )


class GCNSingleArchitecture(GCN):
    def __init__(self, num_node_features, nr_of_layers: int, hidden_channels: int):
        super().__init__()
        self.pool = attention_pooling(num_node_features)

        self.convs = self._return_conv(
            num_node_features, nr_of_layers=nr_of_layers, embeding_size=hidden_channels
        )

        if self.attention:
            input_dim = hidden_channels + num_node_features
        else:
            input_dim = hidden_channels

        self.lins = GCN._return_lin(
            input_dim=input_dim, nr_of_lin_layers=2, embeding_size=hidden_channels,
        )


class GCNPairArchitecture(GCN):
    def __init__(self, num_node_features, nr_of_layers: int, hidden_channels: int):
        super().__init__()

        self.pool = attention_pooling(num_node_features,)

        self.convs_p = self._return_conv(
            num_node_features, nr_of_layers=nr_of_layers, embeding_size=hidden_channels
        )
        self.convs_d = self._return_conv(
            num_node_features, nr_of_layers=nr_of_layers, embeding_size=hidden_channels
        )
        if self.attention:
            input_dim = hidden_channels * 2 + 2 * num_node_features
        else:
            input_dim = hidden_channels * 2

        self.lins = GCN._return_lin(
            input_dim=input_dim, nr_of_lin_layers=2, embeding_size=hidden_channels,
        )

        self.pool = attention_pooling(num_node_features)


class GCNPairArchitectureV2(GCN):
    def __init__(self, num_node_features, nr_of_layers: int, hidden_channels: int):
        super().__init__()

        self.pool = attention_pooling(num_node_features)

        self.convs = self._return_conv(
            num_node_features, nr_of_layers=nr_of_layers, embeding_size=hidden_channels
        )

        if self.attention:
            input_dim = hidden_channels
        else:
            input_dim = hidden_channels

        self.lins_d = GCN._return_lin(
            input_dim=input_dim, nr_of_lin_layers=2, embeding_size=hidden_channels,
        )
        self.lins_p = GCN._return_lin(
            input_dim=input_dim, nr_of_lin_layers=2, embeding_size=hidden_channels,
        )

        self.pool = attention_pooling(num_node_features)


class NNConvPairArchitecture(GCN):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        nr_of_layers: int,
        hidden_channels: int,
    ):
        super().__init__()
        hidden_channels = 16
        self.pool = attention_pooling(num_node_features)

        self.convs_d = GCN._return_nnconv(
            num_node_features,
            num_edge_features,
            nr_of_layers=nr_of_layers,
            embeding_size=hidden_channels,
        )
        self.convs_p = GCN._return_nnconv(
            num_node_features,
            num_edge_features,
            nr_of_layers=nr_of_layers,
            embeding_size=hidden_channels,
        )
        if self.attention:
            input_dim = 2 * hidden_channels + (2 * num_node_features)
        else:
            input_dim = 2 * hidden_channels

        self.lins = GCN._return_lin(
            input_dim=input_dim, nr_of_lin_layers=2, embeding_size=hidden_channels,
        )


#####################################
#####################################
# Combining everything
#####################################
#####################################
class GATProt(GATpKa):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int = 32,
        num_layers: int = 3,
        out_channels=32,
        dropout=0.5,
        attention=False,
    ):
        super().__init__(
            in_channels=num_node_features,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.lins = GATpKa._return_lin(
            input_dim=out_channels, nr_of_lin_layers=2, embeding_size=hidden_channels
        )

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        x_p_batch = data.x_p_batch.to(device=DEVICE)

        x = super().forward(x=x_p, edge_index=data.edge_index_p)
        # global mean pooling
        x = global_mean_pool(x, x_p_batch)  # [batch_size, hidden_channels]
        # run through linear layer
        x = forward_lins(x, self.lins)
        return x


class AttentiveProt(AttentivePka):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int = 32,
        num_layers: int = 3,
        num_timesteps: int = 10,
        out_channels=32,
        dropout=0.5,
        attention=False,
    ):
        super().__init__(
            in_channels=num_node_features,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            edge_dim=num_edge_features,
            num_timesteps=num_timesteps,
        )
        self.lins = AttentivePka._return_lin(
            input_dim=out_channels, nr_of_lin_layers=2, embeding_size=hidden_channels
        )

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        x_p_batch = data.x_p_batch.to(device=DEVICE)

        x = super().forward(
            x=x_p, edge_attr=edge_attr_p, edge_index=data.edge_index_p, batch=x_p_batch
        )
        # global mean pooling
        # x = global_mean_pool(x, x_p_batch)  # [batch_size, hidden_channels]
        # run through linear layer
        x = forward_lins(x, self.lins)
        return x


class GINProt(GINpKa):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int = 32,
        num_layers: int = 3,
        out_channels=32,
        dropout=0.5,
        attention=False,
    ):
        super().__init__(
            in_channels=num_node_features,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.lins = GINpKa._return_lin(
            input_dim=out_channels, nr_of_lin_layers=3, embeding_size=hidden_channels
        )
        self.final_lin = Linear(hidden_channels, 1, device=DEVICE)

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        x_p_batch = data.x_p_batch.to(device=DEVICE)

        x = super().forward(x=x_p, edge_index=data.edge_index_p)
        # global mean pooling
        x = global_mean_pool(x, x_p_batch)  # [batch_size, hidden_channels]
        # run through linear layer
        x = forward_lins(x, self.lins)

        return self.final_lin(F.relu(x))


class GATPair(GATpKa):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int = 32,
        num_layers: int = 3,
        out_channels=32,
        dropout=0.5,
        attention=False,
    ):
        super().__init__(
            in_channels=num_node_features,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.lins = GATpKa._return_lin(
            input_dim=out_channels, nr_of_lin_layers=2, embeding_size=hidden_channels
        )

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        def _forward(x, edge_index, x_batch, func):
            x = func(x=x, edge_index=edge_index)
            # global mean pooling
            x = global_mean_pool(x, x_batch)  # [batch_size, hidden_channels]
            # run through linear layer
            return forward_lins(x, self.lins)

        func = super().forward
        x_p_batch = data.x_p_batch.to(device=DEVICE)
        x_d_batch = data.x_d_batch.to(device=DEVICE)

        x_p = _forward(x_p, data.edge_index_p, x_p_batch, func)
        x_d = _forward(x_d, data.edge_index_d, x_d_batch, func)
        return x_p + x_d


class GINPairV1(GCN):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int = 32,
        num_layers: int = 4,
        out_channels=32,
        dropout=0.5,
        attention=False,
    ):
        super().__init__()

        GIN_p = GINpKa(
            in_channels=num_node_features,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        GIN_d = GINpKa(
            in_channels=num_node_features,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.lins = GINpKa._return_lin(
            input_dim=out_channels * 2,
            nr_of_lin_layers=3,
            embeding_size=hidden_channels,
        )
        self.GIN_p = GIN_p
        self.GIN_d = GIN_d
        self.final_lin = Linear(hidden_channels, 1, device=DEVICE)

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        def _forward(x, edge_index, x_batch, func):
            x = func(x=x, edge_index=edge_index)
            # global mean pooling
            x = global_mean_pool(x, x_batch)  # [batch_size, hidden_channels]
            # run through linear layer
            return x

        x_p_batch = data.x_p_batch.to(device=DEVICE)
        x_d_batch = data.x_d_batch.to(device=DEVICE)

        x_p = _forward(x_p, data.edge_index_p, x_p_batch, self.GIN_p.forward)
        x_d = _forward(x_d, data.edge_index_d, x_d_batch, self.GIN_d.forward)
        x = torch.cat([x_p, x_d], dim=1)
        x = forward_lins(x, self.lins)

        return self.final_lin(F.relu(x))


class GINPairV3(GCN):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int = 32,
        num_layers: int = 4,
        out_channels=32,
        dropout=0.5,
        attention=False,
    ):
        super().__init__()

        GIN_p = GINpKa(
            in_channels=num_node_features,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        GIN_d = GINpKa(
            in_channels=num_node_features,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.lins = GINpKa._return_lin(
            input_dim=out_channels, nr_of_lin_layers=3, embeding_size=hidden_channels,
        )
        self.GIN_p = GIN_p
        self.GIN_d = GIN_d
        self.final_lin = Linear(hidden_channels * 2, 1, device=DEVICE)

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        def _forward(x, edge_index, x_batch, func):
            x = func(x=x, edge_index=edge_index)
            # global mean pooling
            x = global_mean_pool(x, x_batch)  # [batch_size, hidden_channels]
            # run through linear layer
            return forward_lins(x, self.lins)

        x_p_batch = data.x_p_batch.to(device=DEVICE)
        x_d_batch = data.x_d_batch.to(device=DEVICE)

        x_p = _forward(x_p, data.edge_index_p, x_p_batch, self.GIN_p.forward)
        x_d = _forward(x_d, data.edge_index_d, x_d_batch, self.GIN_d.forward)
        x = torch.cat([x_p, x_d], dim=1)

        return self.final_lin(F.relu(x))


class GINPairV2(GINpKa):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int = 32,
        num_layers: int = 3,
        out_channels=32,
        dropout=0.5,
        attention=False,
    ):

        super().__init__(
            in_channels=num_node_features,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.lins_d = GINpKa._return_lin(
            input_dim=out_channels, nr_of_lin_layers=3, embeding_size=hidden_channels
        )
        self.lins_p = GINpKa._return_lin(
            input_dim=out_channels, nr_of_lin_layers=3, embeding_size=hidden_channels
        )
        self.final_lin = Linear(2, 1, device=DEVICE)

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        def _forward(x, edge_index, x_batch, func, lins):
            x = func(x=x, edge_index=edge_index)
            # global mean pooling
            x = global_mean_pool(x, x_batch)  # [batch_size, hidden_channels]
            # run through linear layer
            return forward_lins(x, lins)

        x_p_batch = data.x_p_batch.to(device=DEVICE)
        x_d_batch = data.x_d_batch.to(device=DEVICE)

        x_p = _forward(x_p, data.edge_index_p, x_p_batch, super().forward, self.lins_p)
        x_d = _forward(x_d, data.edge_index_d, x_d_batch, super().forward, self.lins_d)
        return self.final_lin(torch.cat([x_p, x_d], dim=1))


class AttentivePairV1(AttentivePka):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int = 32,
        num_layers: int = 3,
        num_timesteps: int = 10,
        out_channels=32,
        dropout=0.5,
        attention=False,
    ):
        super().__init__(
            in_channels=num_node_features,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            edge_dim=num_edge_features,
            num_timesteps=num_timesteps,
        )

        self.AttentivePka_p = AttentivePka(
            in_channels=num_node_features,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            edge_dim=num_edge_features,
            num_timesteps=num_timesteps,
        )
        self.AttentivePka_d = AttentivePka(
            in_channels=num_node_features,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            edge_dim=num_edge_features,
            num_timesteps=num_timesteps,
        )

        self.lins = AttentivePka._return_lin(
            input_dim=out_channels, nr_of_lin_layers=2, embeding_size=hidden_channels
        )

        self.final_lin = Linear(2, 1, device=DEVICE)

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        def _forward(x, edge_attr, edge_index, batch, func):
            x = func(x=x, edge_attr=edge_attr, edge_index=edge_index, batch=batch)
            # run through linear layer
            return forward_lins(x, self.lins)

        x_p_batch = data.x_p_batch.to(device=DEVICE)
        x_d_batch = data.x_d_batch.to(device=DEVICE)

        x_p = _forward(
            x=x_p,
            edge_attr=edge_attr_p,
            edge_index=data.edge_index_p,
            batch=x_p_batch,
            func=self.AttentivePka_p,
        )
        x_d = _forward(
            x=x_d,
            edge_attr=edge_attr_d,
            edge_index=data.edge_index_d,
            batch=x_d_batch,
            func=self.AttentivePka_d,
        )
        return self.final_lin(F.softmax(torch.Tensor([x_p, x_d])))


class AttentivePair(AttentivePka):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int = 32,
        num_layers: int = 3,
        num_timesteps: int = 10,
        out_channels=32,
        dropout=0.5,
        attention=False,
    ):
        super().__init__(
            in_channels=num_node_features,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            edge_dim=num_edge_features,
            num_timesteps=num_timesteps,
        )

        self.lins = AttentivePka._return_lin(
            input_dim=out_channels, nr_of_lin_layers=2, embeding_size=hidden_channels
        )

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        def _forward(x, edge_attr, edge_index, batch, func):
            x = func(x=x, edge_attr=edge_attr, edge_index=edge_index, batch=batch)
            # run through linear layer
            return forward_lins(x, self.lins)

        func = super().forward
        x_p_batch = data.x_p_batch.to(device=DEVICE)
        x_d_batch = data.x_d_batch.to(device=DEVICE)

        x_p = _forward(
            x=x_p,
            edge_attr=edge_attr_p,
            edge_index=data.edge_index_p,
            batch=x_p_batch,
            func=func,
        )
        x_d = _forward(
            x=x_d,
            edge_attr=edge_attr_d,
            edge_index=data.edge_index_d,
            batch=x_d_batch,
            func=func,
        )
        return x_p + x_d


class GCNProt(GCNSingleArchitecture, GCNSingleForward):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        nr_of_layers: int = 3,
        hidden_channels: int = 96,
        attention=False,
    ):
        self.attention = attention
        super().__init__(num_node_features, nr_of_layers, hidden_channels)

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        return self._forward(x_p, data.edge_index_p, data.x_p_batch)


class GCNDeprot(GCNSingleArchitecture, GCNSingleForward):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        nr_of_layers: int = 3,
        hidden_channels: int = 96,
        attention=False,
    ):
        self.attention = attention
        super().__init__(num_node_features, nr_of_layers, hidden_channels)
        self.pool = attention_pooling(num_node_features)

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        return self._forward(x_d, data.edge_index_d, data.x_d_batch)


class NNConvProt(NNConvSingleArchitecture, NNConvSingleForward):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        nr_of_layers: int = 3,
        hidden_channels: int = 96,
        attention: bool = False,
    ):
        self.attention = attention

        super().__init__(
            num_node_features, num_edge_features, nr_of_layers, hidden_channels
        )
        self.pool = attention_pooling(num_node_features)

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        return self._forward(x_p, data.x_p_batch, edge_attr_p, data.edge_index_p)


class NNConvDeprot(NNConvSingleArchitecture, NNConvSingleForward):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        nr_of_layers: int = 3,
        hidden_channels: int = 96,
        attention: bool = False,
    ):
        self.attention = attention
        super().__init__(
            num_node_features, num_edge_features, nr_of_layers, hidden_channels
        )
        self.pool = attention_pooling(num_node_features)

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        return self._forward(x_d, data.x_d_batch, edge_attr_d, data.edge_index_d)


#####################################
# for pairs
#####################################


class GCNPairTwoConv(GCNPairArchitecture, GCNPairTwoConvForward):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        nr_of_layers: int = 3,
        hidden_channels: int = 96,
        attention: bool = False,
    ):
        self.attention = attention
        super().__init__(num_node_features, nr_of_layers, hidden_channels)

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        return self._forward(x_p, x_d, edge_attr_p, edge_attr_d, data)


class GCNPairSingleConv(GCNPairArchitectureV2, GCNPairOneConvForward):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        nr_of_layers: int = 3,
        hidden_channels: int = 96,
        attention: bool = False,
    ):
        self.attention = attention
        super().__init__(num_node_features, nr_of_layers, hidden_channels)

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        return self._forward(x_p, x_d, edge_attr_p, edge_attr_d, data)


class NNConvPair(NNConvPairArchitecture, NNConvPairForward):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        nr_of_layers: int = 3,
        hidden_channels: int = 96,
        attention: bool = False,
    ):
        self.attention = attention
        super().__init__(
            num_node_features, num_edge_features, nr_of_layers, hidden_channels
        )

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        return self._forward(x_p, x_d, edge_attr_p, edge_attr_d, data)


#####################################
#####################################
#####################################
#####################################
# Functions for training and testing of GCN models

calculate_mse = torch.nn.MSELoss()
calculate_mae = torch.nn.L1Loss()  # that's the MAE Loss


def gcn_train(model, training_loader, optimizer, reg_loader=None):
    model.train()
    if reg_loader:
        for train_data, reg_data in zip(
            training_loader, reg_loader
        ):  # Iterate in batches over the training dataset.
            train_data.to(device=DEVICE)
            out = model(
                x_p=train_data.x_p,
                x_d=train_data.x_d,
                edge_attr_p=train_data.edge_attr_p,
                edge_attr_d=train_data.edge_attr_d,
                data=train_data,
            )
            ref = train_data.reference_value
            loss = calculate_mse(out.flatten(), ref)  # Compute the loss.
            reg_data.to(device=DEVICE)
            out = model(
                x_p=reg_data.x_p,
                x_d=reg_data.x_d,
                edge_attr_p=reg_data.edge_attr_p,
                edge_attr_d=reg_data.edge_attr_d,
                data=reg_data,
            )
            ref = reg_data.reference_value
            loss += calculate_mse(out.flatten(), ref)  # Compute the loss.

            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
    else:
        for data in training_loader:  # Iterate in batches over the training dataset.
            data.to(device=DEVICE)
            out = model(
                x_p=data.x_p,
                x_d=data.x_d,
                edge_attr_p=data.edge_attr_p,
                edge_attr_d=data.edge_attr_d,
                data=data,
            )
            ref = data.reference_value
            loss = calculate_mse(out.flatten(), ref)  # Compute the loss.

            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.


def gcn_test(model, loader) -> float:
    model.eval()
    loss = torch.Tensor([0]).to(device=DEVICE)
    for data in loader:  # Iterate in batches over the training dataset.
        data.to(device=DEVICE)
        out = model(
            x_p=data.x_p,
            x_d=data.x_d,
            edge_attr_p=data.edge_attr_p,
            edge_attr_d=data.edge_attr_d,
            data=data,
        )  # Perform a single forward pass.
        ref = data.reference_value
        loss += calculate_mae(out.flatten(), ref).detach()
    return round(
        float(loss / len(loader)), 3
    )  # MAE loss of batches can be summed and divided by the number of batches


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    all_validation_loss: list,
    validation_loss: float,
    path: str,
    prefix: str,
):
    performance = model.checkpoint
    torch.save(
        {
            "epoch": performance["epoch"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": validation_loss,
        },
        f"{path}/{prefix}model_at_{epoch}.pt",
    )

    # save performance of best model evaluated on validation set
    if epoch != 0:
        if validation_loss < min(all_validation_loss[:-1]):
            torch.save(
                {
                    "epoch": performance["epoch"],
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": validation_loss,
                },
                f"{path}/{prefix}best_model.pt",
            )


def gcn_full_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    path: str = "",
    NUM_EPOCHS: int = 1_000,
    prefix="",
    reg_loader=None,
) -> dict:
    """Training routine

    Args:
        model ([type]): [description]
        train_loader ([type]): [description]
        val_loader ([type]): [description]
        optimizer ([type]): [description]
        path (str, optional): [description]. Defaults to "".
        NUM_EPOCHS (int, optional): [description]. Defaults to 1_000.
        prefix (str, optional): [description]. Defaults to "".
        reg_loader ([type], optional): [description]. Defaults to None.

    Returns:
        dict: [description]
    """
    from torch import optim

    pbar = tqdm(range(model.checkpoint["epoch"], NUM_EPOCHS + 1), desc="Epoch: ")
    results = {}
    results["training-set"] = []
    results["validation-set"] = []
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=150, verbose=True, factor=0.5
    )

    for epoch in pbar:
        if epoch != 0:
            gcn_train(model, train_loader, optimizer, reg_loader)
        if epoch % 5 == 0:
            train_loss = gcn_test(model, train_loader)
            val_loss = gcn_test(model, val_loader)
            pbar.set_description(
                f"Train MAE: {train_loss:.4f}, Validation MAE: {val_loss:.4f}"
            )
            results["training-set"].append(train_loss)
            results["validation-set"].append(val_loss)
            if path:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    results["validation-set"],
                    val_loss,
                    path,
                    prefix,
                )
        scheduler.step(val_loss)

    return results
