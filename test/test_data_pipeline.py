from functional_residue.data.structures import (
    fetch_pdb,
    fetch_alphafold_prediction,
    get_chain_sequence,
    get_standard_residues,
)
from functional_residue.data.graphs import get_residue_distance_mat
from functional_residue.data.embeddings import EmbeddingSet
from functional_residue.models.GAT import GAT
from functional_residue.data.datasets import (
    data_from_structure,
    ProteinStructureDataset,
)
from torch_geometric.nn import GATConv
import pytest
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from pathlib import Path
import tempfile
import shutil


def test_fetch_pdb():
    """
    Test the fetch_pdb function.
    """
    # Fetch a PDB file and save it to the specified directory
    structure = fetch_pdb("101M", ".test_data", return_structure=True)
    assert structure is not None
    assert structure.id == "101M"


def test_get_pdb_distance_mat():
    """
    Test the get_pdb_distance_mat function.
    """
    # Fetch a PDB file and save it to the specified directory
    structure = fetch_pdb("101M", ".test_data", return_structure=True)
    assert structure is not None
    assert structure.id == "101M"
    chain = structure[0]["A"]
    residues = get_standard_residues(chain)
    assert len(residues) == 154
    distance_matrix, residues = get_residue_distance_mat(chain)
    assert distance_matrix.shape == (len(residues), len(residues))


def test_get_residue_distance_mat(**kwargs):
    # structure = fetch_pdb('101M', '.test_data', return_structure=True)
    structure = fetch_alphafold_prediction(
        "P02185", ".test_data", return_structure=True
    )
    assert structure is not None
    assert structure.id == "AF-P02185-F1-model_v4"
    chain = structure[0]["A"]
    distance_matrix, residues = get_residue_distance_mat(chain, **kwargs)
    assert distance_matrix.shape == (len(residues), len(residues))
    x0 = [-16.647, 7.731, 6.117]
    x1 = [-15.617, 6.938, 9.723]
    expected_distance = np.linalg.norm(np.array(x0) - np.array(x1))
    assert distance_matrix[0, 1] == pytest.approx(expected_distance)
    assert distance_matrix[0, 0] == pytest.approx(0)


def test_get_embedding():
    structure = fetch_alphafold_prediction(
        "P02185", ".test_data", return_structure=True
    )
    chain = structure[0]["A"]  # type: ignore
    sequence = get_chain_sequence(chain)
    embedding_set = EmbeddingSet()
    embedding_out = embedding_set.get_many_embeddings(
        sequences=[
            sequence,
        ],
        ids=[
            structure.id,  # type: ignore
        ],
    )
    assert embedding_out[0].shape == (len(sequence), 1024)
    assert len(embedding_out) == 1
    seq = "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG"
    embedding_out_2 = embedding_set.get_many_embeddings(
        sequences=[
            seq,
        ],
        ids=[
            "P02185",
        ],
    )
    assert embedding_out_2[0].shape == (len(sequence), 1024)
    assert (embedding_out[0] == embedding_out[0]).all()


def test_forward():
    gat = GAT(input_dim=1024, hidden_dim=2048, output_dim=1, num_heads=16)
    file_path = (
        Path(__file__).parent.parent
        / "data"
        / "cat_models"
        / "gnn_prott5_xin_only_1.pt"
    )
    if not file_path.exists():
        raise FileNotFoundError(
            f"Model file not found. Please download the model from {file_path}"
        )
    # Load the model weights
    gat.load_state_dict(
        torch.load(file_path, weights_only=True, map_location=torch.device("cpu"))
    )
    assert isinstance(gat, GAT)
    assert isinstance(gat.gat1, GATConv)
    assert isinstance(gat.gat_out, GATConv)

    structure = fetch_alphafold_prediction(
        "P02185", ".test_data", return_structure=True
    )
    chain = structure[0]["A"]  # type: ignore
    sequence = get_chain_sequence(chain)
    embedding_set = EmbeddingSet()
    embedding_out = embedding_set.get_many_embeddings(
        sequences=[
            sequence,
        ],
        ids=[
            structure.id,  # type: ignore
        ],
    )
    assert embedding_out[0].shape == (len(sequence), 1024)
    distance_matrix, residues = get_residue_distance_mat(chain)
    assert distance_matrix.shape == (len(residues), len(residues))
    edge_index = dense_to_sparse(torch.Tensor(distance_matrix < 6.0))[0]
    output = (
        gat(
            torch.Tensor(embedding_out[0]),
            edge_index,
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert isinstance(output, np.ndarray)
    assert output.shape == (len(sequence), 1)


def test_data_creation():
    structure = fetch_alphafold_prediction(
        "P02185", ".test_data", return_structure=True
    )
    data = data_from_structure(structure)  # type: ignore
    assert data.x.size() == (154, 1024)  # type: ignore


def test_dataset_creation():
    structures = [
        fetch_alphafold_prediction("P02185", ".test_data", return_structure=True),
        fetch_alphafold_prediction("P02163", ".test_data", return_structure=True),
    ]
    # temp_data_dir = tempfile.TemporaryDirectory()
    temp_data_dir = Path(".test_data/")
    temp_data_dir.mkdir(parents=True, exist_ok=True)

    for structure in structures:
        data = data_from_structure(structure)  # type: ignore
        torch.save(data, temp_data_dir.name + f"/{structure.id}.pt")  # type: ignore
    dataset = ProteinStructureDataset(temp_data_dir.name)
    assert len(dataset) == len(structures)
    assert dataset[0].x.size() == (154, 1024)  # type: ignore
    shutil.rmtree(temp_data_dir, ignore_errors=True)


if __name__ == "__main__":
    test_dataset_creation()
    test_forward()
    test_get_pdb_distance_mat()
    test_fetch_pdb()
    test_get_embedding()
    test_get_residue_distance_mat(processes=4)
    test_data_creation()
    print("all tests passed")
