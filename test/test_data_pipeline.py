from functional_residue.data.structures import fetch_pdb, fetch_alphafold_prediction
from functional_residue.data.graphs import get_residue_distance_mat
from functional_residue.data.embeddings import EmbeddingSet
import pytest
import numpy as np
from Bio.PDB.Polypeptide import protein_letters_3to1_extended


def test_fetch_pdb():
    """
    Test the fetch_pdb function.
    """
    # Fetch a PDB file and save it to the specified directory
    structure = fetch_pdb("101M", ".test_data", return_structure=True)
    assert structure is not None
    assert structure.id == "101M"


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
    residues = list(chain.get_residues())
    res_names = [
        protein_letters_3to1_extended[residue.resname.upper()] for residue in residues
    ]
    sequence = "".join(res_names)
    embedding_set = EmbeddingSet()
    embedding_out = embedding_set.get_many_embeddings(
        sequences=[
            sequence,
        ],
        ids=[
            structure.id,
        ],
    )
    assert embedding_out.shape == (1, 1024)
    seq = "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG"
    embedding_out_2 = embedding_set.get_many_embeddings(
        sequences=[
            seq,
        ],
        ids=[
            "P02185",
        ],
    )
    assert embedding_out_2.shape == (1, 1024)
    assert (embedding_out == embedding_out).all()


if __name__ == "__main__":
    test_get_embedding()
    test_get_residue_distance_mat(processes=4)
    print("all tests passed")
