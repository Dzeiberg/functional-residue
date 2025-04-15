import urllib
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure
import json
from typing import Optional


def download_file(url, save_file):
    """
    Download a file from a URL and save it to the specified location.

    Parameters:
    url (str): The URL of the file to download.
    save_file (str): The local path where the file will be saved.

    Returns:
    None
    """
    if Path(save_file).exists():
        return
    urllib.request.urlretrieve(url, save_file)


def fetch_pdb(pdb_id: str, save_dir: str | Path, **kwargs) -> Optional[Structure]:
    """
    Fetch a PDB file from the RCSB PDB database and save it to the specified save_dir.

    Parameters:
    pdb_id (str): The PDB ID of the structure to fetch.
    save_dir (str): The local directory to which the PDB file will be saved.

    Optional Parameters:
    - return_structure (bool): If True, return the structure object otherwise return None, default is False.

    Returns:
    None | PDB.Structure.Structure
    """
    return_structure = kwargs.get("return_structure", False)
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{pdb_id}.pdb"
    download_file(url, path)
    if return_structure:
        structure = parse_pdb_structure(pdb_id, path)
        return structure
    return None


def parse_pdb_structure(pdb_id: str, path: str | Path) -> Structure:
    """
    Parse a PDB file and return the structure object.

    Parameters:
    pdb_id (str): The PDB ID of the structure.
    path (str|Path): The path to the PDB file.

    Returns:
    Structure: The parsed structure object.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, path)
    return structure


def fetch_alphafold_prediction(
    uniprot_acc: str, save_dir: str | Path, **kwargs
) -> Optional[Structure]:
    """
    Fetch an AlphaFold prediction from the AlphaFold database.

    Parameters:
    - uniprot_acc (str): The UniProt accession number of the protein.
    - save_dir (str): The local directory to which the AlphaFold prediction will be saved.

    Optional Parameters:
    - return_structure (bool): If True, return the structure object otherwise return None, default is False.

    Returns:
    str: The URL to the AlphaFold prediction.
    """
    query_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_acc}"
    response = urllib.request.urlopen(query_url)
    if response.status != 200:
        raise ValueError(
            f"Error fetching AlphaFold prediction for {uniprot_acc}: {response.status}"
        )
    data = json.loads(response.read().decode("utf-8"))
    if not len(data):
        raise ValueError(f"No AlphaFold prediction found for {uniprot_acc}")
    pdb_url = data[0]["pdbUrl"]
    file_name = Path(pdb_url).stem
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    download_file(pdb_url, save_dir / file_name)
    if kwargs.get("return_structure", False):
        structure = parse_pdb_structure(file_name, save_dir / file_name)
        return structure
