import os
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from Bio.PDB.Chain import Chain
from torch_geometric.utils import dense_to_sparse
import numpy as np
from numpy import ndarray

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
        return torch.load(data_path, weights_only=False)


def data_from_chain(
    chain: Chain, pos_resnums: list | ndarray | None = None, **kwargs
) -> Data:
    """
    Convert a Bio.PDB chain object to a PyTorch Geometric Data object, containing the adjacency matrix and node embeddings features

    Parameters:
    chain (chain): The Bio.PDB chain object to convert.

    Optional:
    - C_alpha_threshold (float): The threshold for the distance between C-alpha atoms to consider when drawing edges; default 6.0
    Returns:
    Data: The converted PyTorch Geometric Data object.
    """
    sequence = chain.seq  # type: ignore
    # Get the node embeddings
    embedding_set = EmbeddingSet()
    chain_id = chain.full_id[0] + "_" + chain.full_id[-1]  # type: ignore
    embedding_out = embedding_set.get_many_embeddings(
        sequences=[
            sequence,
        ],
        ids=[
            chain_id,  # type: ignore
        ],
    )
    # Convert the embedding to a tensor
    embedding_out = torch.tensor(embedding_out[0], dtype=torch.float)
    # Get the C-alpha distance matrix
    distance_matrix, residues = get_residue_distance_mat(chain)
    # Convert the distance matrix to a sparse adjacency matrix
    C_alpha_threshold = kwargs.get("C_alpha_threshold", 6.0)
    edge_index = dense_to_sparse(torch.Tensor(distance_matrix < C_alpha_threshold))[0]
    resnum2idx = {residue.get_id()[1]: idx for idx, residue in enumerate(residues)}
    if pos_resnums is not None:
        pos_resnums = np.array(list(map(resnum2idx.get, pos_resnums)))
        node_labels = torch.zeros(len(residues), dtype=torch.float)  # type: ignore
        node_labels[pos_resnums] = 1
        data = Data(x=embedding_out, edge_index=edge_index, y=node_labels)  # type: ignore
    else:
        data = Data(x=embedding_out, edge_index=edge_index)
    data["sequence"] = sequence
    return data
