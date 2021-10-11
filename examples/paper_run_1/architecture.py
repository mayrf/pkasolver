from config import *

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU, ModuleList
from torch_geometric.nn import GCNConv, NNConv, BatchNorm, global_max_pool
import copy

def convs(edge_conv):
    if edge_conv:
        nn = Seq(Linear(num_edge_features, 16), ReLU(), Linear(16, num_node_features*HIDDEN_CHANNELS))
        nn1 = Seq(Linear(num_edge_features, 16), ReLU(), Linear(16, HIDDEN_CHANNELS* HIDDEN_CHANNELS))
        convs = ModuleList([NNConv(num_node_features, HIDDEN_CHANNELS, nn=nn)])
        convs.extend([NNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS, nn=nn1) for i in range(NUM_GRAPH_LAYERS-1)])
    else:
        convs = ModuleList([GCNConv(num_node_features, HIDDEN_CHANNELS)])
        convs.extend([GCNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS) for i in range(NUM_GRAPH_LAYERS-1)])
    return convs

def lins(paired):
    channels= HIDDEN_CHANNELS
    if paired:
        channels= HIDDEN_CHANNELS*2
#     lins= ModuleList([Seq(Linear(channels, channels), ReLU()) for i in range(NUM_LINEAR_LAYERS-1)])
    lins= ModuleList([Linear(channels, channels) for i in range(NUM_LINEAR_LAYERS-1)])
    lins.extend([Linear(channels, 1)])
    return lins


class GCN(torch.nn.Module):
    def __init__(self, edge_conv):
        super().__init__()
        torch.manual_seed(SEED)
        
        self.convs_x = convs(edge_conv)
        if self.paired:
            self.convs_x2 = convs(edge_conv)
        self.lin = lins(self.paired)
        self.edge_conv = edge_conv
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
        
        
    def forward_helper(self, x, x2, edge_attr, edge_attr2, data):
        
        if self.edge_conv:
            for i in range(len(self.convs_x)):
                x = self.convs_x[i](x, data.edge_index, edge_attr)
                x = x.relu()
            if self.paired:
                for i in range(len(self.convs_x2)):
                    x2 = self.convs_x2[i](x2, data.edge_index2, edge_attr2)
                    x2 = x2.relu()
                
        else:
            for i in range(len(self.convs_x)):
                x = self.convs_x[i](x, data.edge_index)
                x = x.relu()
            if self.paired:
                for i in range(len(self.convs_x2)):
                    x2 = self.convs_x2[i](x2, data.edge_index2)
                    x2 = x2.relu()

        # 2. Readout layer
        x = global_max_pool(x, data.batch.to(device=DEVICE))  # [batch_size, hidden_channels]
        if self.paired: 
            x2 = global_max_pool(x2, data.x2_batch.to(device=DEVICE))
            
        if self.paired:
            x = torch.cat((x, x2), 1)

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        for i in range(len(self.lin)):
                x = self.lin[i](x)
        return x
    
    
    def forward(self, x, x2, edge_attr, edge_attr2, data):
        return self.forward_helper(x, x2, edge_attr, edge_attr2, data)


class GCN_prot(GCN):    
    def __init__(self, edge_conv):
        self.paired = False
        super().__init__(edge_conv)
        
    def forward(self, x, edge_attr, x2, edge_attr2, data):
        return self.forward_helper(x, x2, edge_attr, edge_attr2, data)
    
class GCN_deprot(GCN):    
    def __init__(self, edge_conv):
        self.paired = False
        super().__init__(edge_conv)
        
    def forward(self, x2, edge_attr2, x, edge_attr, data):
        mod = copy.deepcopy(data)
        mod.batch, mod.x2_batch = mod.x2_batch, mod.batch
        mod.edge_index, mod.edge_index2 = mod.edge_index2, mod.edge_index
        return self.forward_helper(x2, x, edge_attr2, edge_attr, mod)
    
class GCN_paired(GCN):
    def __init__(self, edge_conv):
        self.paired = True
        super().__init__(edge_conv)
        
    def forward(self, x, x2, edge_attr, edge_attr2, data):
        return self.forward_helper(x, x2, edge_attr, edge_attr2, data)

def first_check(model,optimizer):
    return {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_data': [train_data[i].ID for i in range(len(train_data))],
        'val_data': [val_data[i].ID for i in range(len(val_data))],
        'node_features': NODE_FEATURES,
        'edge_features': EDGE_FEATURES,
        'progress':'',
        'train_test_split': TRAIN_TEST_SPLIT,
        'num_epochs': NUM_EPOCHS,
        'device': DEVICE,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'paired': PAIRED,
        'num_graph_layers': NUM_GRAPH_LAYERS,
        'num_linear_layers': NUM_LINEAR_LAYERS,
        'hidden_channels': HIDDEN_CHANNELS,
        'best_loss': best_loss
    }




criterion = torch.nn.MSELoss()
criterion_v = torch.nn.L1Loss() # that's the MAE Loss
import pickle

def gcn_train(model,loader, optimizer, device='cpu'):
    model.train()
    for data in loader:  # Iterate in batches over the training dataset.
        data.to(device=device)
        out = model(x=data.x, x2=data.x2,edge_attr=data.edge_attr, edge_attr2=data.edge_attr2, data=data)
        loss = criterion(out.flatten(), data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad() # Clear gradients.
        
def gcn_test(model,loader, device='cpu'):
    model.eval()
    loss = torch.Tensor([0]).to(device=device)
    for data in loader:  # Iterate in batches over the training dataset.
        data.to(device=device)
        out = model(x=data.x, x2=data.x2,edge_attr=data.edge_attr, edge_attr2=data.edge_attr2, data=data)  # Perform a single forward pass.
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
        if epoch % 20 == 0:   #20
            save_checkpoint(model, optimizer, epoch, train_loss, test_loss, path)