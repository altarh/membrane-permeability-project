## Step 1: Convert data into PyTorch Geometric Dataset

import torch
from torch_geometric.data import Data, Dataset


# Custom Dataset class
class CustomGraphDataset(Dataset):
    def __init__(self, graph_triplets, graph_labels=None, transform=None, pre_transform=None):
        super(CustomGraphDataset, self).__init__(transform=transform, pre_transform=pre_transform)
        self.data_list = self._process_examples(graph_triplets,graph_labels=graph_labels)

    def _process_examples(self, graph_triplets,graph_labels=None):
      N = len(graph_triplets)
      data_list = []
      for n in range(N):
          node_features, edge_features, edge_indices = graph_triplets[n]
          label = torch.tensor( graph_labels[n]) if graph_labels is not None else None

          # Create a Data object for each graph
          data = Data(
              x=node_features,               # Node features
              edge_attr=edge_features,       # Edge features
              edge_index=edge_indices.T,       # Edge indices
              y=label                        # Binary label
          )
          data_list.append(data)
      return data_list

    def len(self):
        """
        Return the number of graphs in the dataset.
        """
        return len(self.data_list)

    def get(self, idx):
        """
        Return the Data object at index idx.
        """
        return self.data_list[idx]



from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)


        # Initialize the layers
        self.node_embedding = Linear(dataset.num_node_features, hidden_channels)
        self.conv = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1) # a single continous value for regression


    def forward(self, node_features, edge_features, edge_index, batch):

        # 1. Embed node features
        x = self.node_embedding(node_features)
        x = x.relu()

        # 2. Pass through a [permutation-equivariant] GCN layer

        x = self.conv(x, edge_index) # Element-wise non-linearity
        x = x.relu() # Element-wise non-linearity

        # 3. Global average pooling for obtaining a permutation-invariant representation.
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 4. Apply a final classifier
        x = self.lin(x) # This is the pre-sigmoid output.
        return x


