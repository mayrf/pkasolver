import copy
import pickle

import torch
from torch_geometric.nn.glob import attention
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import (
    GCNConv,
    NNConv,
    global_max_pool,
    global_mean_pool,
    GlobalAttention,
)

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


#####################################
#####################################
# defining GCN for single state
#####################################
#####################################


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        self.checkpoint = {
            "epoch": 0,
            "optimizer_state_dict": "",
            "best_loss": (1, 0),  # NOTE: I don't really understand this option
            "best_states": {},
            "progress_table": {"epoch": [], "train_loss": [], "test_loss": []},
        }


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
        for i in range(len(self.convs)):
            x = x.relu()
            x = self.convs[i](x, edge_index)

        # global max pooling
        x = global_mean_pool(x, x_batch)  # [batch_size, hidden_channels]

        # if attention=True append attention layer
        if self.attention:
            x = torch.cat((x, x_att), 1)

        # set dimensions to zero
        x = F.dropout(x, p=0.5, training=self.training)

        # run through linear layer
        for i in range(len(self.lins)):
            x = self.lins[i](x)
        return x


class GCNPairOneConvForward:
    def _forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        x_p_batch = data.x_p_batch.to(device=DEVICE)
        x_d_batch = data.x_d_batch.to(device=DEVICE)

        # using only a single conv

        for i in range(len(self.convs)):
            x_p = x_p.relu()
            x_p = self.convs[i](x_p, data.edge_index_p)
        for i in range(len(self.convs)):
            x_d = x_d.relu()
            x_d = self.convs[i](x_d, data.edge_index_d)

        x_p = global_mean_pool(x_p, x_p_batch)  # [batch_size, hidden_channels]
        x_d = global_mean_pool(x_d, x_d_batch)

        x_p = F.dropout(x_p, p=0.5, training=self.training)
        x_d = F.dropout(x_d, p=0.5, training=self.training)

        for i in range(len(self.lins)):
            x_p = self.lins[i](x_p)
        for i in range(len(self.lins)):
            x_d = self.lins[i](x_d)

        return x_p + x_d


class GCNPairTwoConvForward:
    def _forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        x_p_batch = data.x_p_batch.to(device=DEVICE)
        x_d_batch = data.x_d_batch.to(device=DEVICE)

        if self.attention:
            x_p_att = self.pool(x_p, x_p_batch)
            x_d_att = self.pool(x_d, x_d_batch)

        for i in range(len(self.convs_p)):
            x_p = x_p.relu()
            x_p = self.convs_p[i](x_p, data.edge_index_p)
        for i in range(len(self.convs_d)):
            x_d = x_d.relu()
            x_d = self.convs_d[i](x_d, data.edge_index_d)

        x_p = global_mean_pool(x_p, x_p_batch)  # [batch_size, hidden_channels]
        x_d = global_mean_pool(x_d, x_d_batch)

        if self.attention:
            x = torch.cat((x_p, x_d, x_p_att, x_d_att), 1)
        else:
            x = torch.cat([x_p, x_d], 1)

        x = F.dropout(x, p=0.5, training=self.training)
        for i in range(len(self.lins)):
            x = self.lins[i](x)
        return x


class NNConvSingleForward:
    def _forward(self, x, x_batch, edge_attr, edge_index):

        x_batch = x_batch.to(device=DEVICE)
        if self.attention:
            x_att = self.pool(x, x_batch)

        for i in range(len(self.convs)):
            x = x.relu()
            x = self.convs[i](x, edge_index, edge_attr)

        x = global_mean_pool(x, x_batch)  # [batch_size, hidden_channels]

        if self.attention:
            x = torch.cat((x, x_att), 1)

        x = F.dropout(x, p=0.5, training=self.training)

        for i in range(len(self.lins)):
            x = self.lins[i](x)
        return x


class NNConvPairForward:
    def _forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):

        x_p_batch, x_d_batch = (
            data.x_p_batch.to(device=DEVICE),
            data.x_d_batch.to(device=DEVICE),
        )
        x_p_att = self.pool(x_p, x_p_batch)
        x_d_att = self.pool(x_d, x_d_batch)

        for i in range(len(self.convs_d)):
            x_p = x_p.relu()
            x_p = self.convs_d[i](x_p, data.edge_index_p, edge_attr_p)

        for i in range(len(self.convs_p)):
            x_d = x_d.relu()
            x_d = self.convs_p[i](x_d, data.edge_index_d, edge_attr_d)

        x_p = global_mean_pool(x_p, x_p_batch)  # [batch_size, hidden_channels]
        x_d = global_mean_pool(x_d, x_d_batch)

        if self.attention:
            x = torch.cat((x_p, x_d, x_p_att, x_d_att), 1)
        else:
            x = torch.cat((x_p, x_d), 1)

        x = F.dropout(x, p=0.5, training=self.training)
        for i in range(len(self.lins)):
            x = self.lins[i](x)
        return x


class NNConvSingleArchitecture(GCN):
    def __init__(self, num_node_features, num_edge_features):
        super().__init__()
        self.pool = attention_pooling(num_node_features)

        self.convs = self._return_nnconv(num_node_features, num_edge_features)

        if self.attention:
            lin1 = Linear(
                16 + num_node_features, 8
            )  # NOTE: adding number of node features
            lin2 = Linear(8, 1)
        else:
            lin1 = Linear(16, 8)
            lin2 = Linear(8, 1)

        self.lins = ModuleList([lin1, lin2])

    @staticmethod
    def _return_nnconv(num_node_features, num_edge_features):
        nn1 = Sequential(
            Linear(num_edge_features, 16), ReLU(), Linear(16, num_node_features * 16),
        )
        nn2 = Sequential(Linear(num_edge_features, 16), ReLU(), Linear(16, 16 * 16),)
        conv1 = NNConv(num_node_features, 16, nn=nn1)
        conv2 = NNConv(16, 16, nn=nn2)
        return ModuleList([conv1, conv2])


class GCNSingleArchitecture(GCN):
    def __init__(self, num_node_features):
        super().__init__()
        self.pool = attention_pooling(num_node_features)

        self.convs = self._return_conv(num_node_features)
        if self.attention:
            lin1 = Linear(16 + num_node_features, 8)
            lin2 = Linear(8, 1)
        else:
            lin1 = Linear(16, 8)
            lin2 = Linear(8, 1)

        self.lins = ModuleList([lin1, lin2])

    @staticmethod
    def _return_conv(num_node_features):
        convs1 = GCNConv(num_node_features, 32)
        convs2 = GCNConv(32, 16)
        convs3 = GCNConv(16, 16)
        return ModuleList([convs1, convs2, convs3])


class GCNPairArchitecture(GCN):
    def __init__(self, num_node_features):
        super().__init__()

        self.pool = attention_pooling(num_node_features)

        self.convs_p = self._return_conv(num_node_features)
        self.convs_d = self._return_conv(num_node_features)

        if self.attention:
            lin1 = Linear(16 * 2 + 2 * num_node_features, 16)
            lin2 = Linear(16, 1)
        else:
            lin1 = Linear(16 * 2, 16)
            lin2 = Linear(16, 1)

        self.lins = ModuleList([lin1, lin2])
        self.pool = attention_pooling(num_node_features)

    @staticmethod
    def _return_conv(num_node_features):
        convs1 = GCNConv(num_node_features, 32)
        convs2 = GCNConv(32, 16)
        convs3 = GCNConv(16, 16)
        return ModuleList([convs1, convs2, convs3])


class GCNPairArchitectureV2(GCN):
    def __init__(self, num_node_features):
        super().__init__()

        self.pool = attention_pooling(num_node_features)

        self.convs = self._return_conv(num_node_features)

        lin1 = Linear(16, 16)
        lin2 = Linear(16, 1)

        self.lins = ModuleList([lin1, lin2])
        self.pool = attention_pooling(num_node_features)

    @staticmethod
    def _return_conv(num_node_features):
        convs1 = GCNConv(num_node_features, 32)
        convs2 = GCNConv(32, 16)
        convs3 = GCNConv(16, 16)
        return ModuleList([convs1, convs2, convs3])


class NNConvPairArchitecture(GCN):
    def __init__(self, num_node_features, num_edge_features):
        super().__init__()

        self.pool = attention_pooling(num_node_features)

        self.convs_d = self._return_nnconv(num_node_features, num_edge_features)
        self.convs_p = self._return_nnconv(num_node_features, num_edge_features)

        if self.attention:
            lin1 = Linear(32 + (2 * num_node_features), 8)
            lin2 = Linear(8, 1)
        else:
            lin1 = Linear(32, 8)
            lin2 = Linear(8, 1)
        self.lins = ModuleList([lin1, lin2])

    @staticmethod
    def _return_nnconv(num_node_features, num_edge_features):
        nn1 = Sequential(
            Linear(num_edge_features, 16), ReLU(), Linear(16, num_node_features * 16),
        )
        nn2 = Sequential(Linear(num_edge_features, 16), ReLU(), Linear(16, 16 * 16),)
        conv1 = NNConv(num_node_features, 16, nn=nn1)
        conv2 = NNConv(16, 16, nn=nn2)
        return ModuleList([conv1, conv2])


#####################################
#####################################
# Architecture
#####################################
#####################################


class GCNProt(GCNSingleArchitecture, GCNSingleForward):
    def __init__(
        self, num_node_features, num_edge_features, attention=False,
    ):
        self.attention = attention
        super().__init__(num_node_features)
        print(f"Attention pooling: {self.attention}")

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        return self._forward(x_p, data.edge_index_p, data.x_p_batch)


class GCNDeprot(GCNSingleArchitecture, GCNSingleForward):
    def __init__(
        self, num_node_features, num_edge_features, attention=False,
    ):
        self.attention = attention
        super().__init__(num_node_features)
        self.pool = attention_pooling(num_node_features)
        print(f"Attention pooling: {self.attention}")

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        return self._forward(x_d, data.edge_index_d, data.x_d_batch)


class NNConvProt(NNConvSingleArchitecture, NNConvSingleForward):
    def __init__(
        self, num_node_features, num_edge_features, attention=False,
    ):
        self.attention = attention
        print(f"Attention pooling: {self.attention}")

        super().__init__(num_node_features, num_edge_features)
        self.pool = attention_pooling(num_node_features)

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        return self._forward(x_p, data.x_p_batch, edge_attr_p, data.edge_index_p)


class NNConvDeprot(NNConvSingleArchitecture, NNConvSingleForward):
    def __init__(
        self, num_node_features, num_edge_features, attention=False,
    ):
        self.attention = attention
        print(f"Attention pooling: {self.attention}")
        super().__init__(num_node_features, num_edge_features)
        self.pool = attention_pooling(num_node_features)

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        return self._forward(x_d, data.x_d_batch, edge_attr_d, data.edge_index_d)


#####################################
# for pairs
#####################################


class GCNPairTwoConv(GCNPairArchitecture, GCNPairTwoConvForward):
    def __init__(
        self, num_node_features: int, num_edge_features: int, attention: bool = False,
    ):
        self.attention = attention
        super().__init__(num_node_features)
        print(f"Attention pooling: {self.attention}")

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        return self._forward(x_p, x_d, edge_attr_p, edge_attr_d, data)


class GCNPairSingleConv(GCNPairArchitectureV2, GCNPairOneConvForward):
    def __init__(
        self, num_node_features: int, num_edge_features: int, attention: bool = False,
    ):
        self.attention = attention
        super().__init__(num_node_features)
        print(f"Attention pooling: {self.attention}")

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        return self._forward(x_p, x_d, edge_attr_p, edge_attr_d, data)


class NNConvPair(NNConvPairArchitecture, NNConvPairForward):
    def __init__(
        self, num_node_features: int, num_edge_features: int, attention: bool = False,
    ):
        self.attention = attention
        super().__init__(num_node_features, num_edge_features)
        print(f"Attention pooling: {self.attention}")

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        return self._forward(x_p, x_d, edge_attr_p, edge_attr_d, data)


#####################################
#####################################
#####################################
#####################################
# Functions for training and testing of GCN models

criterion = torch.nn.MSELoss()
criterion_v = torch.nn.L1Loss()  # that's the MAE Loss


def gcn_train(model, loader, optimizer):
    model.train()
    for data in loader:  # Iterate in batches over the training dataset.
        out = model(
            x_p=data.x_p,
            x_d=data.x_d,
            edge_attr_p=data.edge_attr_p,
            edge_attr_d=data.edge_attr_d,
            data=data,
        )
        loss = criterion(out.flatten(), data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def gcn_test(model, loader):
    model.eval()
    loss = torch.Tensor([0]).to(device=DEVICE)
    for data in loader:  # Iterate in batches over the training dataset.
        out = model(
            x_p=data.x_p,
            x_d=data.x_d,
            edge_attr_p=data.edge_attr_p,
            edge_attr_d=data.edge_attr_d,
            data=data,
        )  # Perform a single forward pass.
        loss += criterion_v(out.flatten(), data.y).detach()
    return round(
        float(loss / len(loader)), 3
    )  # MAE loss of batches can be summed and divided by the number of batches


def save_checkpoint(model, optimizer, epoch, train_loss, test_loss, path):
    point = model.checkpoint
    point["epoch"] = epoch + 1
    if point["best_loss"][0] > test_loss:
        point["best_loss"] = (test_loss, epoch, copy.deepcopy(model.state_dict()))
        point["best_states"][epoch] = (test_loss, copy.deepcopy(model.state_dict()))
    point["optimizer_state"] = optimizer.state_dict()
    point["progress_table"]["epoch"].append(epoch)
    point["progress_table"]["train_loss"].append(train_loss)
    point["progress_table"]["test_loss"].append(test_loss)
    with open(path + "model.pkl", "wb") as pickle_file:
        pickle.dump(model, pickle_file)


def gcn_full_training(
    model, train_loader, val_loader, optimizer, path:str='', NUM_EPOCHS:int = 1_000
) -> dict:
    pbar = tqdm(range(model.checkpoint["epoch"], NUM_EPOCHS + 1), desc="Epoch: ")
    results = {}
    results["training-set"] = []
    results["validation-set"] = []
    for epoch in pbar:
        if epoch != 0:
            gcn_train(model, train_loader, optimizer)
        if epoch % 20 == 0:
            train_loss = gcn_test(model, train_loader)
            val_loss = gcn_test(model, val_loader)
            pbar.set_description(
                f"Train MAE: {train_loss:.4f}, Validation MAE: {val_loss:.4f}"
            )
            results["training-set"].append(train_loss)
            results["validation-set"].append(val_loss)
    if path and epoch % 40 == 0:
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path)

    return results
