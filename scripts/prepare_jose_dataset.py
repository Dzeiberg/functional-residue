import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import sys

sys.path.append(str(Path(__file__).parent.parent))
from functional_residue.data.structures import fetch_pdb, get_chain_resnames
from functional_residue.data.datasets import data_from_chain


def prepare_jose_dataset(task: str, save_directory: str):
    """
    Prepare Jose's functional residue dataset for training.

    Parameters:
    task (str): The name of the task to prepare the dataset for.
    save_dir (str): The directory to save the prepared dataset.
    """
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    structures_dir = save_dir / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = save_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    data_path = (
        Path(__file__).parent.parent
        / "data"
        / "lugomartinez_ploscb_2016_datasets"
        / "structural_and_functional_sites"
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Data path {data_path} does not exist.")
    # Load the dataset
    df = pd.read_csv(
        data_path / f"{task}.pos",
        sep="\t",
        names=["PDB", "Chain", "ResidueNum", "Residue"],
    )
    for (PDB, chain), residues in tqdm(
        df.groupby(["PDB", "Chain"]),
        total=len(df["PDB"].unique()),
        desc="Processing PDBs",
    ):
        # get the structure
        structure = fetch_pdb(PDB, structures_dir, return_structure=True)
        chain = structure[0][chain]  # type: ignore
        chain_resnames = get_chain_resnames(chain)
        res_nums = pd.to_numeric(residues["ResidueNum"]).astype(int).values
        for idx, (_, residue) in enumerate(residues.iterrows()):
            # get the residue number
            res_num = res_nums[idx]
            # get the residue name
            res_name = residue["Residue"]
            # check if the residue number is in the chain
            if chain_resnames[res_num] != res_name:
                raise ValueError(
                    f"Residue name {res_name} does not match the chain sequence {chain_resnames[res_num]} at position {res_num}"
                )
        # initialize node labels
        pos_resnums = np.array(res_nums)
        # get the data
        instance = data_from_chain(chain, pos_resnums=pos_resnums)
        # save the instance
        chain_id = chain.full_id[0] + "_" + chain.full_id[2]  # type: ignore
        torch.save(instance, processed_dir / f"{chain_id}.pt")


if __name__ == "__main__":
    prepare_jose_dataset(
        "Cat",
        "/Users/dz/Documents/research/functional_residue/data/lugomartinez_ploscb_2016_datasets/Cat",
    )
