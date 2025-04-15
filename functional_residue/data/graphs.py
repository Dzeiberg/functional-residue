import numpy as np
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from sklearn.metrics import pairwise_distances
from multiprocessing import Pool
from typing import List, Tuple
import os  # Add import for os to handle file cleanup


def get_residue_coords(res: Residue) -> np.ndarray:
    """Get coordinates of all atoms of a residue
    Args:
        res (Residue): residue object
    Returns:
        coords (np.ndarray): coordinates of all atoms of the residue
    """
    # Get the coordinates of the atoms of a residue
    coords = np.stack([np.array(a.get_vector().get_array()) for a in res.get_atoms()])
    return coords


def get_Ca_coords(res: Residue) -> np.ndarray:
    """
    Get the coordinate of the C-alpha atom of a residue
    Args:
        res (Residue): residue object

    Returns:
        coords (np.ndarray): coordinates of the C-alpha atom of the residue
    """

    # Get the coordinates of the C-alpha atom of a residue
    try:
        coords = np.array(res["CA"].get_vector())
    except KeyError:
        raise KeyError(f"Could not find C-alpha atom for residue {res}")
    return coords


def get_residue_pair_dist(a, b, atom_distances, corresp_res):
    # Distance between two residues
    a_idxs = np.where(corresp_res == a)[0]
    b_idxs = np.where(corresp_res == b)[0]
    return a, b, atom_distances[np.ix_(a_idxs, b_idxs)].min()


def get_residue_distance_mat(
    chain: Chain, **kwargs
) -> Tuple[np.ndarray, List[Residue]]:
    """Get residue distance matrix from pdb file

    Args:
        pdb_file_path (str): path to pdb file
        chain (str): chain to read [A,B,C,...] (usually A)

    Optional Args:
        Ca_only (bool): if True, only use C-alpha atoms for distance calculation (default: True)
        processes (int): number of processes to use for distance calculation (default: os.process_cpu_count())

    Returns:
        residueDistances (np.array) (n_residues x n_residues): residue distance matrix
        residues (list): list of residues
    """
    residues = chain.get_list()
    n_residues = len(residues)
    if kwargs.get("Ca_only", True):
        coord_arrays = [get_Ca_coords(r) for r in residues]
        coords = np.stack(coord_arrays)
        residue_indices = np.arange(len(coord_arrays))
    else:
        coord_arrays = [get_residue_coords(r) for r in residues]
        coords = np.concatenate(coord_arrays)
        residue_indices = np.concatenate(
            [[i] * len(arr) for i, arr in enumerate(coord_arrays)]
        )

    # Temporary file paths
    residue_indices_path = "residue_indices.array"
    atom_distances_path = "atom_distances.array"

    # try:
    mm_residue_indices = np.memmap(
        residue_indices_path, dtype=int, mode="w+", shape=residue_indices.shape
    )
    mm_residue_indices[:] = residue_indices[:]
    atom_distances = pairwise_distances(
        np.array(list(map(lambda v: v._ar, coords))), n_jobs=-1
    )
    mm_atom_dists = np.memmap(
        atom_distances_path, dtype="float32", mode="w+", shape=atom_distances.shape
    )
    mm_atom_dists[:] = atom_distances[:]

    mm_residue_indices = np.memmap(
        residue_indices_path, dtype=int, mode="r", shape=residue_indices.shape
    )
    mm_atom_dists = np.memmap(
        atom_distances_path, dtype="float32", mode="r", shape=atom_distances.shape
    )
    pool = Pool(kwargs.get("processes", os.cpu_count()))
    pairDists = pool.starmap(
        get_residue_pair_dist,
        [
            (a, b, mm_atom_dists, mm_residue_indices)
            for a in range(n_residues)
            for b in range(a + 1, n_residues)
        ],
    )
    pool.close()
    pool.join()
    residueDistances = np.zeros((n_residues, n_residues))
    for p in pairDists:
        residueDistances[p[0], p[1]] = p[2]
        residueDistances[p[1], p[0]] = p[2]
    # finally:
    # Cleanup temporary files
    if os.path.exists(residue_indices_path):
        os.remove(residue_indices_path)
    if os.path.exists(atom_distances_path):
        os.remove(atom_distances_path)

    return residueDistances, residues
