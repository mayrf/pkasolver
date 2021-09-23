from config import *

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU, ModuleList
from torch_geometric.nn import GCNConv, NNConv, BatchNorm, global_max_pool
import copy

# Adding GNN architectures for pka predictions

def nnconv_block(embedding_size, num_graph_layer):
    nn = Seq(Linear(num_edge_features, 16), ReLU(), Linear(16, num_node_features*embedding_size))
    nn1 = Seq(Linear(num_edge_features, 16), ReLU(), Linear(16, embedding_size* embedding_size))
    convs = ModuleList([NNConv(num_node_features, embedding_size, nn=nn)])
    convs.extend([NNConv(embedding_size, embedding_size, nn=nn1) for i in range(num_graph_layer-1)])
    return convs

def gcnconv_block(embedding_size, num_graph_layer):
    convs = ModuleList([GCNConv(num_node_features, embedding_size)])
    convs.extend([GCNConv(embedding_size, embedding_size) for i in range(num_graph_layer-1)])
    return convs

def lin_block(embedding_size, num_linear_layer):
    lins= ModuleList([Linear(embedding_size, embedding_size) for i in range(num_linear_layer-1)])
    lins.extend([Linear(embedding_size, 1)])
    return lins

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        self.checkpoint = {
        'epoch': 0,
        'optimizer_state_dict': '',
        'best_loss': (best_loss,0),
        'best_states':{},    
        'progress_table':{'epoch':[],
                          'train_loss':[],
                          'test_loss':[]
                         }
    }
        
class GCN_prot(GCN):
    def __init__(self, embedding_size, num_graph_layer, num_linear_layer):
        super().__init__()
        self.convs_x = gcnconv_block(embedding_size, num_graph_layer)
        self.lin = lin_block(embedding_size, num_linear_layer)

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        for i in range(len(self.convs_x)):
            x_p = self.convs_x[i](x_p, data.edge_index_p)
            x_p = x_p.relu()
            
        x_p = global_max_pool(x_p, data.x_p_batch.to(device=DEVICE))  # [batch_size, hidden_channels]
        x_p = F.dropout(x_p, p=0.5, training=self.training)

        for i in range(len(self.lin)):
                x_p = self.lin[i](x_p)
        return x_p

class GCN_deprot(GCN):    
    def __init__(self, embedding_size, num_graph_layer, num_linear_layer):
        super().__init__()
        self.convs_x = gcnconv_block(embedding_size, num_graph_layer)
        self.lin = lin_block(embedding_size, num_linear_layer)

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        for i in range(len(self.convs_x)):
            x_d = self.convs_x[i](x_d, data.edge_index_d)
            x_d = x_d.relu()
            
        x_d = global_max_pool(x_d, data.x_d_batch.to(device=DEVICE))  # [batch_size, hidden_channels]
        x_d = F.dropout(x_d, p=0.5, training=self.training)
        
        for i in range(len(self.lin)):
                x_d = self.lin[i](x_d)
        return x_d

class GCN_pair(GCN):
    def __init__(self, embedding_size, num_graph_layer, num_linear_layer):
        super().__init__()
        self.convs_x = gcnconv_block(embedding_size, num_graph_layer)
        self.convs_x2 = gcnconv_block(embedding_size, num_graph_layer)
        self.lin = lin_block(embedding_size*2, num_linear_layer)     

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):                
        for i in range(len(self.convs_x)):
            x_p = self.convs_x[i](x_p, data.edge_index_p)
            x_p = x_p.relu()
        for i in range(len(self.convs_x2)):
            x_d = self.convs_x2[i](x_d, data.edge_index_d)
            x_d = x_d.relu()

        x_p = global_max_pool(x_p, data.x_p_batch.to(device=DEVICE))  # [batch_size, hidden_channels]
        x_d = global_max_pool(x_d, data.x_d_batch.to(device=DEVICE))
        x = torch.cat((x_p, x_d), 1)
        x = F.dropout(x, p=0.5, training=self.training)

        for i in range(len(self.lin)):
                x = self.lin[i](x)
        return x

class NNConv_prot(GCN):
    def __init__(self, embedding_size, num_graph_layer, num_linear_layer):
        super().__init__()
        self.convs_x = nnconv_block(embedding_size, num_graph_layer)
        self.lin = lin_block(embedding_size, num_linear_layer)  

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        for i in range(len(self.convs_x)):
            x_p = self.convs_x[i](x_p, data.edge_index_p, edge_attr_p)
            x_p = x_p.relu()

        x_p = global_max_pool(x_p, data.x_p_batch.to(device=DEVICE))  # [batch_size, hidden_channels]
        x_p = F.dropout(x_p, p=0.5, training=self.training)
        
        for i in range(len(self.lin)):
                x_p = self.lin[i](x_p)
        return x_p

class NNConv_deprot(GCN):
    def __init__(self, embedding_size, num_graph_layer, num_linear_layer):
        super().__init__()  
        self.convs_x = nnconv_block(embedding_size, num_graph_layer)
        self.lin = lin_block(embedding_size, num_linear_layer)  

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        for i in range(len(self.convs_x)):
            x_d = self.convs_x[i](x_d, data.edge_index_d, edge_attr_d)
            x_d = x_d.relu()
                
        x_d = global_max_pool(x_d, data.x_d_batch.to(device=DEVICE))  # [batch_size, hidden_channels]
        x_d = F.dropout(x_d, p=0.5, training=self.training)
        
        for i in range(len(self.lin)):
                x_d = self.lin[i](x_d)
        return x_d
    
class NNConv_pair(GCN):
    def __init__(self, embedding_size, num_graph_layer, num_linear_layer):
        super().__init__()
        self.convs_x = nnconv_block(embedding_size, num_graph_layer)
        self.convs_x2 = nnconv_block(embedding_size, num_graph_layer)
        self.lin = lin_block(embedding_size*2, num_linear_layer)  

    def forward(self, x_p, x_d, edge_attr_p, edge_attr_d, data):
        for i in range(len(self.convs_x)):
            x_p = self.convs_x[i](x_p, data.edge_index_p, edge_attr_p)
            x_p = x_p.relu()

        for i in range(len(self.convs_x2)):
            x_d = self.convs_x2[i](x_d, data.edge_index_d, edge_attr_d)
            x_d = x_d.relu()

        x_p = global_max_pool(x_p, data.x_p_batch.to(device=DEVICE))  # [batch_size, hidden_channels] 
        x_d = global_max_pool(x_d, data.x_d_batch.to(device=DEVICE))
        x = torch.cat((x_p, x_d), 1)
        x = F.dropout(x, p=0.5, training=self.training)
        
        for i in range(len(self.lin)):
                x = self.lin[i](x)
        return x

# Functions for training and testing of GCN models
    
criterion = torch.nn.MSELoss()
criterion_v = torch.nn.L1Loss() # that's the MAE Loss
import pickle

def gcn_train(model,loader, optimizer, device='cpu'):
    model.train()
    for data in loader:  # Iterate in batches over the training dataset.
        data.to(device=device)
        out = model(x_p=data.x_p, x_d=data.x_d,edge_attr_p=data.edge_attr_p, edge_attr_d=data.edge_attr_d, data=data)
        loss = criterion(out.flatten(), data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad() # Clear gradients.

def gcn_test(model,loader, device='cpu'):
    model.eval()
    loss = torch.Tensor([0]).to(device=device)
    for data in loader:  # Iterate in batches over the training dataset.
        data.to(device=device)
        out = model(x_p=data.x_p, x_d=data.x_d,edge_attr_p=data.edge_attr_p, edge_attr_d=data.edge_attr_d, data=data)  # Perform a single forward pass.
        loss += criterion_v(out.flatten(), data.y)
    return round(float(loss/len(loader)),3) # MAE loss of batches can be summed and divided by the number of batches

def save_checkpoint(model, optimizer, epoch, train_loss, test_loss, path):
    point = model.checkpoint
    point['epoch'] = epoch + 1
    if point['best_loss'][0] > test_loss: 
        point['best_loss'] = (test_loss, epoch, copy.deepcopy(model.state_dict()))
        point['best_states'][epoch] = (test_loss, copy.deepcopy(model.state_dict())) 
    point['optimizer_state']=optimizer.state_dict()
    point['progress_table']['epoch'].append(epoch)
    point['progress_table']['train_loss'].append(train_loss)
    point['progress_table']['test_loss'].append(test_loss)
    with open(path+'model.pkl', 'wb') as pickle_file:
            pickle.dump(model,pickle_file)
    

def gcn_full_training(model,train_loader, val_loader, optimizer, path, device='cpu'):
    for epoch in range(model.checkpoint['epoch'], NUM_EPOCHS+1):
        if epoch != 0: 
            gcn_train(model, train_loader, optimizer, device)
        if epoch % 20 == 0:
            train_loss = gcn_test(model, train_loader, device)
            test_loss = gcn_test(model, val_loader, device)
            print(f'Epoch: {epoch:03d}, Train MAE: {train_loss:.4f}, Test MAE: {test_loss:.4f}')
        if epoch % 40 == 0:   #20
            save_checkpoint(model, optimizer, epoch, train_loss, test_loss, path)