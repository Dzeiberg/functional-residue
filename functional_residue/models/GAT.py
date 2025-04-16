import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=16):
        super(GAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat_out = GATConv(
            hidden_dim * num_heads, output_dim, heads=1, concat=False
        )

    def forward(
        self, embedding: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the GAT model.

        Args:
            embeddings (torch.Tensor): Input node features, shape (N_residues, N_features).
            edge_index (torch.Tensor): Edge indices, CSR format, shape (2, N_edges).

        Returns:
            torch.Tensor: Output node features after GAT layers, shape (N_residues, output_dim).
        """
        embedding = torch.relu(self.gat1(embedding, edge_index))
        output = self.gat_out(embedding, edge_index)
        return output
