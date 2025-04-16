# Functional Residue Analysis

This package provides tools for analyzing functional residues in protein structures using embeddings, graph representations, and a Graph Attention Network (GAT) model.

## Installation

To install the package, clone the repository and install the dependencies:

```bash
git clone https://github.com/Dzeiberg/functional-residue.git
cd functional_residue
pip install -r requirements.txt
```

## Fetching a Structure

To fetch a protein structure, use the following command:

```python
from functional_residue.data.structures import fetch_pdb, fetch_alphafold_prediction

# Example: Fetch structure by PDB ID
structure = fetch_pdb("101M",'structures/',return_structure=True)
# Example: Fetch AlphaFold structure
structure = fetch_alphafold_prediction("P02185", "structures/", return_structure=True)
```

## Obtaining Embeddings

Generate embeddings for a protein:

```python
from functional_residue.data.embeddings import EmbeddingSet

sequences = [ "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG",
"MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASEDLKKHGATVLTALGGILKKKGQHEAQLKPLAQSHATKHKIPVKYLEFISEVIIQVLQSKHPGDFGADAQGAMGKALELFRNDIAAKYKELGFQG"]
uniprot_accessions = ["P02185","P02163"]
embedding_set = EmbeddingSet()
protein_embeddings = embedding_set.get_many_embeddings(
    sequences=sequences,
    ids=uniprot_accessions
)
```

## Obtaining a Graph from the Structure

Convert the protein structure into a graph representation:

```python
from functional_residue.data.structures import fetch_alphafold_prediction
from functional_residue.data.graphs import get_residue_distance_mat

structure = fetch_alphafold_prediction("P02185", "structures/", return_structure=True)
chain = structure[0]["A"]
distance_matrix, residues = get_residue_distance_mat(chain)
# Draw edges between all resiudes with Ca atoms closer than 6Å
Ca_edge_threshold = 6
graph = distance_matrix < Ca_edge_threshold
```

## License

This project is licensed under the  MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.
