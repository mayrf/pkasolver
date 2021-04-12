import random
from torch_geometric.data import DataLoader
import torch

#Train/Test Split PyG Datasets

def pyg_split(dataset,train_test_split, shuffle=True):
    """Take List of PyG Data oojcts and a split ratio between 0 and 1 
    and return a list of Training data and a list of test data.
    """
    if shuffle:
        random.shuffle(dataset)
    
    split_length=int(len(dataset)*train_test_split)
    train_dataset = dataset[:split_length]
    test_dataset = dataset[split_length:]
    return train_dataset, test_dataset


#PyG Dataset to Dataloader 
def dataset_to_dataloader(data, batch_size, shuffle=False):
    """Take a PyG Dataset and return a Dataloader object.
    
    batch_size must be defined.
    Optional shuffle can be enabled.
    """
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, follow_batch=['x', 'x2'])

# PyG Dataset Split an send to loader
def pyg_split_to_loaders(dataset,train_test_split, batch_size, shuffle=False):
    """Take a PyG Dataset and split ratio between 0 and 1 
    and return train_loader and test_loader.  
    """
    if shuffle:
        random.shuffle(dataset)
    
    if train_test_split == 0 or train_test_split == 1:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, follow_batch=['x', 'x2'])
    
    else:
        split_length=int(len(dataset)*train_test_split)
        train_dataset = dataset[:split_length]
        test_dataset = dataset[split_length:]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, follow_batch=['x', 'x2'])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, follow_batch=['x', 'x2'])
        return train_loader, test_loader
    
def update_checkpoint(checkpoint,epoch,model, optimizer, update, checkpoint_path):
    """Take checkpoint, epoch, model, optimizer, update string and checkpoint path.
    Save checkpoint and return checkpoint object.
    """
    checkpoint['epoch']=epoch
    checkpoint['model_state']=model
    checkpoint['optimizer_state']=optimizer
    checkpoint['progress']+= update + '\n'
    torch.save(checkpoint,checkpoint_path)
    return checkpoint
