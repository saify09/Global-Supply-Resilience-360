import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch_geometric.transforms as T
from torch_geometric.data import Data

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.encoder = GraphSAGE(in_channels, hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encoder(x, edge_index)
        return self.decode(z, edge_label_index)

    def decode(self, z, edge_label_index):
        # Dot product of node embeddings to predict edge existence
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

def train_link_prediction(x, edge_index, epochs=10):
    # Prepare Data
    data = Data(x=x, edge_index=edge_index)
    
    # Split edges
    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.1, num_test=0.1,
        is_undirected=False, 
        add_negative_train_samples=True
    )(data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LinkPredictor(x.size(1), 64, 32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f"Training GNN on {device}...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward Pass
        # We use train_data.edge_index for message passing
        # And train_data.edge_label_index for supervision
        z = model.encoder(train_data.x.to(device), train_data.edge_index.to(device))
        
        out = model.decode(z, train_data.edge_label_index.to(device))
        loss = criterion(out, train_data.edge_label.to(device))
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    return model

if __name__ == "__main__":
    # Test stub
    x = torch.randn(100, 16)
    edge_index = torch.randint(0, 100, (2, 500))
    train_link_prediction(x, edge_index, epochs=5)
