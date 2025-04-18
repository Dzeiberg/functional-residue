import os
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from Bio.PDB.Structure import Structure
from torch_geometric.utils import dense_to_sparse
from numpy import ndarray

from functional_residue.data.structures import get_chain_sequence
from functional_residue.data.graphs import get_residue_distance_mat
from functional_residue.data.embeddings import EmbeddingSet


class ProteinStructureDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.graph_files = [
            f for f in os.listdir(self.processed_dir) if f.endswith(".pt")
        ]

    @property
    def processed_dir(self):
        return self.root

    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        data_path = os.path.join(self.processed_dir, self.graph_files[idx])
        return torch.load(data_path)


def data_from_structure(
    structure: Structure, node_labels: ndarray | None = None, **kwargs
) -> Data:
    """
    Convert a Bio.PDB Structure object to a PyTorch Geometric Data object, containing the adjacency matrix and node embeddings features

    Parameters:
    structure (Structure): The Bio.PDB Structure object to convert.

    Optional:
    - C_alpha_threshold (float): The threshold for the distance between C-alpha atoms to consider when drawing edges; default 6.0
    Returns:
    Data: The converted PyTorch Geometric Data object.
    """
    chain = structure[0]["A"]  # type: ignore
    sequence = get_chain_sequence(chain)
    # Get the node embeddings
    embedding_set = EmbeddingSet()
    embedding_out = embedding_set.get_many_embeddings(
        sequences=[
            sequence,
        ],
        ids=[
            structure.id,  # type: ignore
        ],
    )
    # Convert the embedding to a tensor
    embedding_out = torch.tensor(embedding_out[0], dtype=torch.float)
    # Get the C-alpha distance matrix
    distance_matrix, residues = get_residue_distance_mat(chain)
    # Convert the distance matrix to a sparse adjacency matrix
    C_alpha_threshold = kwargs.get("C_alpha_threshold", 6.0)
    edge_index = dense_to_sparse(torch.Tensor(distance_matrix < C_alpha_threshold))[0]
    if node_labels is not None:
        node_labels = torch.tensor(node_labels, dtype=torch.float)  # type: ignore
        assert node_labels.shape[0] == embedding_out.shape[0], "Node labels must match the number of nodes in the graph"  # type: ignore
        data = Data(x=embedding_out, edge_index=edge_index, y=node_labels)  # type: ignore
    else:
        data = Data(x=embedding_out, edge_index=edge_index)
    data["sequence"] = sequence
    return data
