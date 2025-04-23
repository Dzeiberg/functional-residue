from functional_residue.models.GAT import GAT
import torch
from tqdm import tqdm
from typing import List
from torch_geometric.data import Data
import numpy as np
from pathlib import Path

from functional_residue.data.structures import parse_pdb_structure
from functional_residue.data.datasets import data_from_chain


class InferencePipeline(object):
    def __init__(self, model_checkpoint_filepaths: List[str], **kwargs):
        """
        Initialize the InferencePipeline with the model checkpoint filepaths.

        Parameters:
        model_checkpoint_filepaths (List[str]): List of paths to the model checkpoint files.

        Optional Parameters:
        - gat_input_dim (int): Input dimension for the GAT model, default is 1024.
        - gat_hidden_dim (int): Hidden dimension for the GAT model, default is 2048.
        - gat_output_dim (int): Output dimension for the GAT model, default is 1.
        - gat_num_heads (int): Number of attention heads for the GAT model, default is 16.
        """
        self.gat_input_dim = kwargs.get("gat_input_dim", 1024)
        self.gat_hidden_dim = kwargs.get("gat_hidden_dim", 2048)
        self.gat_output_dim = kwargs.get("gat_output_dim", 1)
        self.gat_num_heads = kwargs.get("gat_num_heads", 16)
        self.model_checkpoints = []
        for model_checkpoint_filepath in model_checkpoint_filepaths:
            self.model_checkpoints.append(
                torch.load(
                    model_checkpoint_filepath,
                    weights_only=True,
                    map_location=torch.device("cpu"),
                )
            )
        self.models = []
        for model_checkpoint in self.model_checkpoints:
            model = GAT(
                input_dim=self.gat_input_dim,
                hidden_dim=self.gat_hidden_dim,
                output_dim=self.gat_output_dim,
                num_heads=self.gat_num_heads,
            )
            model.load_state_dict(model_checkpoint)
            self.models.append(model)

    def infer_on_instance(self, instance: Data) -> np.ndarray:
        """
        Run inference on a single instance using the loaded GAT models.

        Parameters:
        instance (Data): The PyTorch Geometric Data object to run inference on.

        Returns:
        List[torch.Tensor]: List of output tensors from each model.
        """
        outputs = []
        for model in self.models:
            with torch.no_grad():
                output = model(instance.x, instance.edge_index)
                outputs.append(output)
        # sigmoid and average the outputs from all models
        outputs = torch.stack(outputs)
        outputs = torch.sigmoid(outputs)
        outputs = torch.mean(outputs, dim=0)
        # convert to numpy
        outputs = outputs.numpy()
        return outputs

    def predict(self, pdb_filepath: str, chain_id: str) -> np.ndarray:
        """
        Run inference on a PDB file and return the predictions.

        Parameters:
        pdb_filepath (str): Path to the PDB file.

        Returns:
        np.ndarray: The predictions for the PDB file.
        """
        protein_id = Path(pdb_filepath).stem
        # Load the PDB file and convert it to a PyTorch Geometric Data object
        structure = parse_pdb_structure(protein_id, pdb_filepath)
        chain = structure[0][chain_id]  # type: ignore
        if not hasattr(chain, "seq"):
            raise ValueError(
                f"Chain {chain.id} does not have a sequence attribute, only {chain.__dict__.keys()} is available."
            )
        data = data_from_chain(chain)
        # Run inference on the single PDB file
        return self.infer_on_instance(data)
