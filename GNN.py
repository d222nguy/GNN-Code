import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torch_geometric
from torch_geometric.utils import to_dense_adj

dataset = torch_geometric.datasets.Planetoid(root = 'Cora', name = 'Cora')
print('Number of graphs: {0}'.format(len(dataset)))
print('Number of features: ', dataset.num_features)
print('Number of classes: ', dataset.num_classes)

data = dataset[0]
print('Data: ', data)
print('Data.edge_index: ', data.edge_index) # Tensor 2*|E|
print('Number of nodes: ', data.num_nodes)
print('Number of edges: ', data.num_edges)
print('Average node degree: ', data.num_edges/data.num_nodes)
print('Train mask: ', data.train_mask)
print('Number of training examples: ', sum(list(data.train_mask)))
print('Number of validation examples:  ', sum(list(data.val_mask)))
print('Number of test examples:  ', sum(list(data.test_mask)))
print('Number of features', data.x.shape[1])

print('Corresponding adjacency matrix: ', to_dense_adj(data.edge_index))
print('Adjacency shape: ', to_dense_adj(data.edge_index).shape)
print('data.x.shape[0]', data.x.shape[0])
def train(model, data, num_epochs, use_edge_index = False):
    if not use_edge_index:
        #Create adjacency matrix from edge_index
        adj = to_dense_adj(data.edge_index)[0] + torch.eye(data.x.shape[0])
    else:
        adj = data.edge_index
    #Set up loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)

    def get_acc(outs, y, mask):
        return (outs[mask].argmax(dim = 1) == y[mask]).sum().float() / mask.sum()
    
    best_acc_val = -1
    for epoch in range(num_epochs):
        #Zero grad --> Forward --> Backward 
        optimizer.zero_grad()
        outs = model(data.x, adj)
        loss = loss_fn(outs[data.train_mask], data.y[data.train_mask])
        loss.backward() #Calculate gradient
        optimizer.step() #Parameter update

        #Compute accuracies
        acc_val = get_acc(outs, data.y, data.val_mask)
        acc_test = get_acc(outs, data.y, data.test_mask)
        if acc_val > best_acc_val:
            best_acc_val = acc_val 
            print('[Epoch {0}/{1}] Loss: {2} | Val: {3} | Test: {4}'.format(epoch + 1, num_epochs, loss, acc_val, acc_test))

class MLPModel(nn.Module):
    def __init__(self, num_layers = 2, sz_in = 1433, sz_hid = 32, sz_out = 7):
        super().__init__()

        self.layers = nn.ModuleList([nn.Linear(
            sz_in if i == 0 else sz_hid,
            sz_out if i == num_layers - 1 else sz_hid
        ) for i in range(num_layers)])
    
    def forward(self, fts, adj):
        for i in range(len(self.layers)):
            fts = self.layers[i](fts)
            if i < len(self.layers) - 1:
                fts = torch.relu(fts)
        return fts 

class GCNModel(nn.Module):
    def __init__(self, num_layers = 2, sz_in = 1433, sz_hid = 32, sz_out = 7):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(
            sz_in if i == 0 else sz_hid, 
            sz_out if i == num_layers - 1 else sz_hid
        ) for i in range(num_layers)])
    def forward(self, fts, adj):
        deg = adj.sum(axis = 1, keepdim = True) #(N, 1)
        for i in range(len(self.layers)):
            fts = self.layers[i](fts) #HW
            fts = adj @ fts / deg
            # print(fts.shape)
            if i < len(self.layers) - 1:
                fts = torch.relu(fts)
        return fts
# train(GCNModel(), data, num_epochs = 100)


#Equivalent, PyTorch Geometric
import torch_geometric.nn as geo_nn
### BEGIN SOLUTION
model = geo_nn.Sequential('x, edge_index', [
    (geo_nn.GCNConv(1433, 32), 'x, edge_index -> x'),
    nn.ReLU(inplace=True),
    (geo_nn.GCNConv(32, 32), 'x, edge_index -> x'),
])
train(model, data, num_epochs=100, use_edge_index=True)


