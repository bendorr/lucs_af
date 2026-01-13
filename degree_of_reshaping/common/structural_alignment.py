"""
Structural alignment of pdb files using the Kabsch algorithm with NumPy.

Author: Ben Orr
Date: 10.22.24

This module provides functions for structural alignment of pdb files using the
Kabsch algorithm implemented with NumPy. It uses BioPandas for pdb file I/O
and manipulation.

The module supports:
- Alignment based on C-alpha atoms
- Automatic handling of structures with different lengths
- RMSD calculation after alignment
- Export of aligned structures to PDB format

Key Functions
-------------
kabsch : Core Kabsch algorithm for optimal rotation/translation
apply_transform : Apply rotation and translation to coordinates
rmsd_ : Calculate RMSD after Kabsch alignment
pdb_align : Align two PDB structures and calculate RMSD
align_and_dump_pdb : Align structures and save aligned pdb file
coord_extractor : Extract coordinates from PandasPdb dataframes

References
----------
Kabsch, W. (1976). "A solution for the best rotation to relate two sets of vectors".
Acta Crystallographica Section A. 32 (5): 922–923.
"""

from typing import Tuple, List, Optional, Any
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain


def apply_transform(
    A: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """
    Apply rotation and translation transformation to coordinates.

    Parameters
    ----------
    A : np.ndarray
        Input coordinates with shape [batch_size, n_atoms, 3].
    R : np.ndarray
        Rotation matrix with shape [batch_size, 3, 3].
    t : np.ndarray
        Translation vector with shape [batch_size, 1, 3].

    Returns
    -------
    np.ndarray
        Transformed coordinates: A_aligned = R * A^T + t

    Notes
    -----
    This function applies the transformation: A_aligned = R * A + t
    where R is a rotation matrix and t is a translation vector.

    Examples
    --------
    >>> A = np.random.randn(1, 10, 3)
    >>> R = np.eye(3)[np.newaxis, :, :]  # Identity rotation
    >>> t = np.array([[[1.0, 2.0, 3.0]]])  # Translation
    >>> A_transformed = apply_transform(A, R, t)
    """
    A_aligned = np.matmul(R, A.transpose(0, 2, 1)).transpose(0, 2, 1) + t
    return A_aligned


def kabsch(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Kabsch algorithm to find optimal rotation and translation.

    The Kabsch algorithm computes the optimal rotation matrix and translation vector
    to align coordinate set A onto coordinate set B, minimizing the RMSD between them.

    Parameters
    ----------
    A : np.ndarray
        Source coordinate set with shape [batch_size, n_atoms, 3].
        These coordinates will be aligned to B.
    B : np.ndarray
        Target coordinate set with shape [batch_size, n_atoms, 3].
        A will be aligned to match these coordinates.

    Returns
    -------
    A_aligned : np.ndarray
        Aligned coordinates of set A with shape [batch_size, n_atoms, 3].
        These are the transformed coordinates after applying rotation R and translation t.
    R : np.ndarray
        Rotation matrix with shape [batch_size, 3, 3].
        The optimal rotation to align A onto B.
    t : np.ndarray
        Translation vector with shape [batch_size, 1, 3].
        The translation to apply after rotation.

    Notes
    -----
    The algorithm follows these steps:
    1. Center both coordinate sets at their centroids
    2. Compute the covariance matrix H = A_c^T * B_c
    3. Perform SVD: H = U * S * V^T
    4. Compute optimal rotation: R = V * U^T
    5. Compute translation: t = b_mean - R * a_mean
    6. Apply transformation: A_aligned = R * A + t

    This is the core Kabsch algorithm that minimizes the RMSD between
    two paired sets of coordinates through optimal rigid-body transformation.

    See Also
    --------
    apply_transform : Applies the computed R and t to coordinates
    rmsd_ : Calculates RMSD after performing Kabsch alignment

    Examples
    --------
    >>> import numpy as np
    >>> # Create two sets of 10 atoms in 3D space
    >>> A = np.random.randn(1, 10, 3)
    >>> B = np.random.randn(1, 10, 3)
    >>> A_aligned, R, t = kabsch(A, B)
    >>> # A_aligned is now optimally aligned to B
    """
    # Compute centroids of both coordinate sets
    # Using double precision for numerical stability
    a_mean = A.mean(axis=1, keepdims=True).astype(np.float64)
    b_mean = B.mean(axis=1, keepdims=True).astype(np.float64)

    # Center both coordinate sets at the origin
    A_c = A - a_mean
    B_c = B - b_mean

    # Compute covariance matrix H = A_c^T * B_c
    # This matrix encodes the relationship between the two centered coordinate sets
    H = np.matmul(A_c.transpose(0, 2, 1), B_c)  # Shape: [batch_size, 3, 3]

    # Perform Singular Value Decomposition: H = U * S * V^T
    # The optimal rotation can be computed from U and V
    U, S, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)

    # Compute optimal rotation matrix: R = V * U^T
    # This rotation minimizes the RMSD between A and B
    R = np.matmul(V, U.transpose(0, 2, 1))  # Shape: [batch_size, 3, 3]

    # Compute optimal translation vector: t = b_mean - R * a_mean
    # This translates the rotated A to align with B's centroid
    t = b_mean - np.matmul(R, a_mean.transpose(0, 2, 1)).transpose(0, 2, 1)

    # Apply the transformation: A_aligned = R * A^T + t
    A_aligned = apply_transform(A, R, t)

    return A_aligned, R, t


def coord_extractor(
    structure: Structure,
    chainids: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract coordinates from a BioPython Structure object.

    This function extracts both C-alpha coordinates (for alignment) and
    all-atom coordinates from a PDB structure.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython Structure object containing coordinate information.
    chainids : Optional[List[str]], optional
        List of chain IDs to extract. If None, extracts all chains.

    Returns
    -------
    coords : np.ndarray
        C-alpha coordinates with shape [1, n_residues, 3].
    all_atom_coords : np.ndarray
        All atom coordinates with shape [1, n_atoms, 3].

    Notes
    -----
    This function first extracts all atom coordinates, then filters for
    C-alpha atoms only (used for structural alignment). If specific chain
    IDs are provided, only those chains are included in the C-alpha extraction.

    Examples
    --------
    >>> from Bio.PDB import PDBParser
    >>> parser = PDBParser(QUIET=True)
    >>> structure = parser.get_structure('protein', 'structure.pdb')
    >>> ca_coords, all_coords = coord_extractor(structure)
    >>> print(ca_coords.shape)  # [1, n_residues, 3]
    """
    # Get the first model
    model = structure[0]
    
    # Extract all atom coordinates first
    all_atom_list = []
    for chain in model:
        for residue in chain:
            for atom in residue:
                all_atom_list.append(atom.get_coord())
    all_atom_coords = np.array(all_atom_list)[np.newaxis, :, :]
    
    # Extract C-alpha coordinates
    ca_list = []
    for chain in model:
        # Filter by chain IDs if specified
        if chainids is not None and chain.id not in chainids:
            continue
        for residue in chain:
            if 'CA' in residue:
                ca_list.append(residue['CA'].get_coord())
    coords = np.array(ca_list)[np.newaxis, :, :]

    return coords, all_atom_coords


def rmsd_(
    pred: np.ndarray,
    true: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Calculate RMSD after Kabsch alignment.

    Aligns pred to true using the Kabsch algorithm and calculates the
    root mean squared deviation (RMSD) between them.

    Parameters
    ----------
    pred : np.ndarray
        Predicted/source coordinates to be aligned, shape [batch_size, n_atoms, 3].
    true : np.ndarray
        True/target coordinates, shape [batch_size, n_atoms, 3].

    Returns
    -------
    pred_aligned : np.ndarray
        Aligned pred coordinates, shape [batch_size, n_atoms, 3].
    rmsd : float
        RMSD value after alignment.
    R : np.ndarray
        Rotation matrix, shape [batch_size, 3, 3].
    t : np.ndarray
        Translation vector, shape [batch_size, 1, 3].

    Notes
    -----
    RMSD is calculated as: sqrt(mean(sum((pred_aligned - true)^2)))
    This provides a measure of structural similarity after optimal alignment.
    Lower RMSD values indicate better structural agreement.

    Examples
    --------
    >>> pred = np.random.randn(1, 100, 3)
    >>> true = np.random.randn(1, 100, 3)
    >>> aligned, rmsd_val, R, t = rmsd_(pred, true)
    >>> print(f"RMSD: {rmsd_val:.2f} Å")
    """
    # Perform Kabsch alignment
    pred_aligned, R, t = kabsch(pred.astype(np.float64), true.astype(np.float64))

    # Calculate RMSD: sqrt(mean(sum(squared_distances)))
    rmsd = np.mean(np.sqrt(np.sum((pred_aligned - true)**2, -1)), -1)
    
    # Convert to scalar if it's an array
    if isinstance(rmsd, np.ndarray):
        rmsd = float(rmsd.item())

    return pred_aligned, rmsd, R, t


def pdb_align(
    pdb1: str,
    pdb2: str,
    chainids1: Optional[List[str]] = None,
    chainids2: Optional[List[str]] = None
) -> Tuple[np.ndarray, float, Structure, np.ndarray, np.ndarray, np.ndarray]:
    """
    Align two pdb structures and calculate RMSD.

    This function loads two pdb files, extracts C-alpha coordinates,
    aligns pdb2 to pdb1 using the Kabsch algorithm, and calculates RMSD.
    If the structures have different lengths, they are cropped to match.

    Parameters
    ----------
    pdb1 : str
        Path to the first (reference) pdb file.
    pdb2 : str
        Path to the second pdb file to be aligned.
    chainids1 : Optional[List[str]], optional
        Chain IDs to use from pdb1. If None, uses all chains.
    chainids2 : Optional[List[str]], optional
        Chain IDs to use from pdb2. If None, uses all chains.

    Returns
    -------
    aligned_struc : np.ndarray
        Aligned C-alpha coordinates of pdb2, shape [n_residues, 3].
    rmsd : float
        RMSD value after alignment.
    structure2 : Bio.PDB.Structure.Structure
        BioPython Structure object for pdb2 (useful for saving aligned structure).
    R : np.ndarray
        Rotation matrix used for alignment.
    t : np.ndarray
        Translation vector used for alignment.
    all_atom_coords2 : np.ndarray
        All atom coordinates of pdb2 before alignment.

    Notes
    -----
    If the structures have different numbers of residues, they are cropped
    to the length of the shorter structure for alignment. This ensures that
    the alignment uses corresponding residues.

    Examples
    --------
    >>> aligned, rmsd, structure, R, t, all_coords = pdb_align("ref.pdb", "model.pdb")
    >>> print(f"RMSD: {rmsd:.2f} Å")
    >>> print(f"Aligned structure shape: {aligned.shape}")
    """
    # Load pdb files
    parser = PDBParser(QUIET=True)
    structure1 = parser.get_structure('ref', pdb1)
    structure2 = parser.get_structure('mobile', pdb2)

    # Extract coordinates
    coord1, all_atom_coords1 = coord_extractor(structure1, chainids=chainids1)
    coord2, all_atom_coords2 = coord_extractor(structure2, chainids=chainids2)

    # Determine minimum length (handle structures with different lengths)
    coord1_length = coord1.shape[1]
    coord2_length = coord2.shape[1]
    min_length = min(coord1_length, coord2_length)

    # Crop coordinates to matching length
    cropped_coord1 = coord1[:, :min_length, :]
    cropped_coord2 = coord2[:, :min_length, :]

    # Perform alignment and calculate RMSD
    # Note: coord2 is aligned to coord1
    aligned_struc, rmsd, R, t = rmsd_(cropped_coord2, cropped_coord1)

    return aligned_struc.squeeze(), rmsd, structure2, R, t, all_atom_coords2


def align_and_dump_pdb(
    pdb1: str,
    pdb2: str,
    chainids1: Optional[List[str]] = None,
    chainids2: Optional[List[str]] = None
) -> float:
    """
    Align two pdb structures and save the aligned structure.

    This function aligns pdb2 to pdb1 using C-alpha atoms, then applies
    the transformation to all atoms of pdb2 and saves the aligned structure
    to a new pdb file.

    Parameters
    ----------
    pdb1 : str
        Path to the first (reference) pdb file.
    pdb2 : str
        Path to the second pdb file to be aligned.
    chainids1 : Optional[List[str]], optional
        Chain IDs to use from pdb1. If None, uses all chains.
    chainids2 : Optional[List[str]], optional
        Chain IDs to use from pdb2. If None, uses all chains.

    Returns
    -------
    float
        RMSD value after alignment.

    Notes
    -----
    The aligned structure is saved to the same directory as pdb2 with
    the prefix "aa_aligned_" (all-atom aligned). All atoms are transformed
    and saved.

    The transformation (R, t) computed from C-alpha alignment is applied
    to all atoms to maintain structural integrity.

    Examples
    --------
    >>> rmsd = align_and_dump_pdb("reference.pdb", "model.pdb")
    All-atom PDB destination /path/to/aa_aligned_model.pdb
    >>> print(f"RMSD: {rmsd:.2f} Å")
    RMSD: 2.45 Å
    """
    # Perform alignment
    aligned_struc, rmsd, structure2, R, t, all_atom_coords2 = pdb_align(
        pdb1, pdb2, chainids1, chainids2
    )

    # Apply transformation to all atoms
    all_atom_aligned = apply_transform(all_atom_coords2.astype(np.float64), R, t)
    all_atom_aligned = all_atom_aligned.squeeze()

    # Apply transformation to all atoms in the structure
    atom_idx = 0
    model = structure2[0]
    for chain in model:
        for residue in chain:
            for atom in residue:
                atom.set_coord(all_atom_aligned[atom_idx])
                atom_idx += 1

    # Construct output path
    aa_aligned_dest = "/".join(pdb2.split('/')[:-1]) + "/aa_aligned_" + pdb2.split('/')[-1]

    # Save aligned structure
    print(f"All-atom PDB destination {aa_aligned_dest}")
    io = PDBIO()
    io.set_structure(structure2)
    io.save(aa_aligned_dest)

    return rmsd
