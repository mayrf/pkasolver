from .test_data_processing import generate_pairwise_data
from ..model import pyg_split
from torch_geometric.data import DataLoader
from torch.nn import Linear, Module
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool

# Model Class:
class Net(Module):
    def __init__(self, hidden_channels, num_features):
        super(Net, self).__init__()

        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)

        self.fc1 = Linear(hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def test_splitting():
    training, test = pyg_split([e for e in range(100)], 0.8)
    assert (len(training)) == 80
    assert (len(test)) == 20

    training, test = pyg_split([e for e in range(1000)], 0.8)
    assert (len(training)) == 800
    assert (len(test)) == 200


def setup_data_loader():
    train_test_split = 0.8
    batch_size = 32

    dataset = generate_pairwise_data()
    train_data, test_data = pyg_split(dataset, train_test_split)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def initialize_gcn(device: str):
    train_loader, test_loader = setup_data_loader()
    num_features = train_loader.dataset[0].m1.num_features
    negative_net, neutral_net, positive_net = (
        Net(hidden_channels=32, num_features=num_features).to(device),
        Net(hidden_channels=32, num_features=num_features).to(device),
        Net(hidden_channels=32, num_features=num_features).to(device),
    )
    return negative_net, neutral_net, positive_net


def test_initialize_gcn():
    negative_net, neutral_net, positive_net = initialize_gcn(device="cpu")
    # initialize_gcn(device="cuda")


def test_setup_data_loader():
    train_loader, test_loader = setup_data_loader()
    print(train_loader)


def evaluate_train_set(loader, net, criterion, optimizer):
    net.train()
    for data in loader:  # Iterate in batches over the training dataset.
        out = net(
            data.x, data.edge_attr, data.edge_index, data.x_batch
        )  # Perform a single forward pass.
        loss = criterion(out.flatten(), data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def evaluate_test_set(loader, net, criterion, device):
    net.eval()
    loss = torch.Tensor([0]).to(device=device)
    for data in loader:  # Iterate in batches over the training dataset.
        out = net(
            data.x, data.edge_attr, data.edge_index, data.x_batch
        )  # Perform a single forward pass.
        loss += criterion(out.flatten(), data.y)
    return loss / len(loader)  # MAE loss of batches can be summed and div


def test_training():
    import torch

    learning_rate = 0.001
    num_epochs = 20
    device = "cpu"

    train_loader, test_loader = setup_data_loader()
    negative_net, neutral_net, positive_net = initialize_gcn(device=device)
    params = (
        list(negative_net.parameters())
        + list(neutral_net.parameters())
        + list(positive_net.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    criterion = torch.nn.MSELoss()
    criterion_v = torch.nn.L1Loss()  # that's the MAE Loss
    # for epoch in range(10):
    #     train_acc = evaluate_train_set(train_loader, net, criterion, optimizer)
    #     # test_acc = evaluate_test_set(test_loader)
