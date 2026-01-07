"""
Structural alignment of pdb files using the Kabsch algorithm with PyTorch.

Author: Ben Orr
Date: 10.22.24

This module provides functions for structural alignment of pdb files using the
Kabsch algorithm implemented with PyTorch. It uses BioPandas for pdb file I/O
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
import torch
from biopandas.pdb import PandasPdb
import torch.nn.functional as F
import pandas


def apply_transform(
    A: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotation and translation transformation to coordinates.

    Parameters
    ----------
    A : torch.Tensor
        Input coordinates with shape [batch_size, n_atoms, 3].
    R : torch.Tensor
        Rotation matrix with shape [batch_size, 3, 3].
    t : torch.Tensor
        Translation vector with shape [batch_size, 1, 3].

    Returns
    -------
    torch.Tensor
        Transformed coordinates: A_aligned = R * A^T + t

    Notes
    -----
    This function applies the transformation: A_aligned = R * A + t
    where R is a rotation matrix and t is a translation vector.

    Examples
    --------
    >>> A = torch.randn(1, 10, 3)
    >>> R = torch.eye(3).unsqueeze(0)  # Identity rotation
    >>> t = torch.tensor([[[1.0, 2.0, 3.0]]])  # Translation
    >>> A_transformed = apply_transform(A, R, t)
    """
    A_aligned = torch.bmm(R, A.transpose(1, 2)).transpose(1, 2) + t
    return A_aligned


def kabsch(A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform Kabsch algorithm to find optimal rotation and translation.

    The Kabsch algorithm computes the optimal rotation matrix and translation vector
    to align coordinate set A onto coordinate set B, minimizing the RMSD between them.

    Parameters
    ----------
    A : torch.Tensor
        Source coordinate set with shape [batch_size, n_atoms, 3].
        These coordinates will be aligned to B.
    B : torch.Tensor
        Target coordinate set with shape [batch_size, n_atoms, 3].
        A will be aligned to match these coordinates.

    Returns
    -------
    A_aligned : torch.Tensor
        Aligned coordinates of set A with shape [batch_size, n_atoms, 3].
        These are the transformed coordinates after applying rotation R and translation t.
    R : torch.Tensor
        Rotation matrix with shape [batch_size, 3, 3].
        The optimal rotation to align A onto B.
    t : torch.Tensor
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
    >>> import torch
    >>> # Create two sets of 10 atoms in 3D space
    >>> A = torch.randn(1, 10, 3)
    >>> B = torch.randn(1, 10, 3)
    >>> A_aligned, R, t = kabsch(A, B)
    >>> # A_aligned is now optimally aligned to B
    """
    # Compute centroids of both coordinate sets
    # Using double precision for numerical stability
    a_mean = A.mean(dim=1, keepdims=True).type('torch.DoubleTensor')
    b_mean = B.mean(dim=1, keepdims=True).type('torch.DoubleTensor')

    # Center both coordinate sets at the origin
    A_c = A - a_mean
    B_c = B - b_mean

    # Compute covariance matrix H = A_c^T * B_c
    # This matrix encodes the relationship between the two centered coordinate sets
    H = torch.bmm(A_c.transpose(1, 2), B_c)  # Shape: [batch_size, 3, 3]

    # Perform Singular Value Decomposition: H = U * S * V^T
    # The optimal rotation can be computed from U and V
    U, S, V = torch.svd(H)

    # Compute optimal rotation matrix: R = V * U^T
    # This rotation minimizes the RMSD between A and B
    R = torch.bmm(V, U.transpose(1, 2))  # Shape: [batch_size, 3, 3]

    # Compute optimal translation vector: t = b_mean - R * a_mean
    # This translates the rotated A to align with B's centroid
    t = b_mean - torch.bmm(R, a_mean.transpose(1, 2)).transpose(1, 2)

    # Apply the transformation: A_aligned = R * A^T + t
    A_aligned = apply_transform(A, R, t)

    return A_aligned, R, t


def coord_extractor(
    df: pandas.DataFrame,
    chainids: Optional[List[str]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract coordinates from a PandasPdb ATOM dataframe.

    This function extracts both C-alpha coordinates (for alignment) and
    all-atom coordinates from a PDB ATOM dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        PandasPdb ATOM dataframe containing coordinate information.
    chainids : Optional[List[str]], optional
        List of chain IDs to extract. If None, extracts all chains.

    Returns
    -------
    coords : torch.Tensor
        C-alpha coordinates with shape [1, n_residues, 3].
    all_atom_coords : torch.Tensor
        All atom coordinates with shape [1, n_atoms, 3].

    Notes
    -----
    This function first extracts all atom coordinates, then filters for
    C-alpha atoms only (used for structural alignment). If specific chain
    IDs are provided, only those chains are included in the C-alpha extraction.

    Examples
    --------
    >>> from biopandas.pdb import PandasPdb
    >>> ppdb = PandasPdb().read_pdb('structure.pdb')
    >>> ca_coords, all_coords = coord_extractor(ppdb.df['ATOM'])
    >>> print(ca_coords.shape)  # [1, n_residues, 3]
    """
    # Extract all atom coordinates first
    X = df['x_coord'].tolist()
    Y = df['y_coord'].tolist()
    Z = df['z_coord'].tolist()
    all_atom_coords = torch.tensor([X, Y, Z]).transpose(0, 1).unsqueeze(0)

    # Filter by chain IDs if specified
    if chainids is not None:
        df = df[df['chain_id'].isin(chainids)]

    # Extract C-alpha coordinates only
    df = df[df['atom_name'] == 'CA']
    X = df['x_coord'].tolist()
    Y = df['y_coord'].tolist()
    Z = df['z_coord'].tolist()
    coords = torch.tensor([X, Y, Z]).transpose(0, 1).unsqueeze(0)

    return coords, all_atom_coords


def rmsd_(
    pred: torch.Tensor,
    true: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate RMSD after Kabsch alignment.

    Aligns pred to true using the Kabsch algorithm and calculates the
    root mean squared deviation (RMSD) between them.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted/source coordinates to be aligned, shape [batch_size, n_atoms, 3].
    true : torch.Tensor
        True/target coordinates, shape [batch_size, n_atoms, 3].

    Returns
    -------
    pred_aligned : torch.Tensor
        Aligned pred coordinates, shape [batch_size, n_atoms, 3].
    rmsd : torch.Tensor
        RMSD value after alignment, shape [batch_size].
    R : torch.Tensor
        Rotation matrix, shape [batch_size, 3, 3].
    t : torch.Tensor
        Translation vector, shape [batch_size, 1, 3].

    Notes
    -----
    RMSD is calculated as: sqrt(mean(sum((pred_aligned - true)^2)))
    This provides a measure of structural similarity after optimal alignment.
    Lower RMSD values indicate better structural agreement.

    Examples
    --------
    >>> pred = torch.randn(1, 100, 3)
    >>> true = torch.randn(1, 100, 3)
    >>> aligned, rmsd_val, R, t = rmsd_(pred, true)
    >>> print(f"RMSD: {rmsd_val.item():.2f} Å")
    """
    # Perform Kabsch alignment
    pred_aligned, R, t = kabsch(pred.type(torch.float64), true.type(torch.float64))

    # Calculate RMSD: sqrt(mean(sum(squared_distances)))
    rmsd = torch.mean(torch.sqrt(torch.sum((pred_aligned - true).pow(2), -1)), -1)

    return pred_aligned, rmsd, R, t


def pdb_align(
    pdb1: str,
    pdb2: str,
    chainids1: Optional[List[str]] = None,
    chainids2: Optional[List[str]] = None
) -> Tuple[torch.Tensor, torch.Tensor, PandasPdb, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    aligned_struc : torch.Tensor
        Aligned C-alpha coordinates of pdb2, shape [n_residues, 3].
    rmsd : torch.Tensor
        RMSD value after alignment.
    ppdb2 : PandasPdb
        PandasPdb object for pdb2 (useful for saving aligned structure).
    R : torch.Tensor
        Rotation matrix used for alignment.
    t : torch.Tensor
        Translation vector used for alignment.
    all_atom_coords2 : torch.Tensor
        All atom coordinates of pdb2 before alignment.

    Notes
    -----
    If the structures have different numbers of residues, they are cropped
    to the length of the shorter structure for alignment. This ensures that
    the alignment uses corresponding residues.

    Examples
    --------
    >>> aligned, rmsd, ppdb, R, t, all_coords = pdb_align("ref.pdb", "model.pdb")
    >>> print(f"RMSD: {rmsd.item():.2f} Å")
    >>> print(f"Aligned structure shape: {aligned.shape}")
    """
    # Load pdb files
    ppdb1 = PandasPdb().read_pdb(pdb1)
    ppdb2 = PandasPdb().read_pdb(pdb2)

    # Extract coordinates
    coord1, all_atom_coords1 = coord_extractor(ppdb1.df['ATOM'], chainids=chainids1)
    coord2, all_atom_coords2 = coord_extractor(ppdb2.df['ATOM'], chainids=chainids2)

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

    return aligned_struc.squeeze(), rmsd, ppdb2, R, t, all_atom_coords2


def align_and_dump_pdb(
    pdb1: str,
    pdb2: str,
    chainids1: Optional[List[str]] = None,
    chainids2: Optional[List[str]] = None
) -> torch.Tensor:
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
    torch.Tensor
        RMSD value after alignment.

    Notes
    -----
    The aligned structure is saved to the same directory as pdb2 with
    the prefix "aa_aligned_" (all-atom aligned). Only C-alpha atoms are
    included in the output file.

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
    aligned_struc, rmsd, ppdb2, R, t, all_atom_coords2 = pdb_align(
        pdb1, pdb2, chainids1, chainids2
    )

    # Apply transformation to all atoms
    all_atom_aligned = apply_transform(all_atom_coords2.type(torch.float64), R, t)
    all_atom_aligned = all_atom_aligned.squeeze()

    # Create new PandasPdb object with aligned coordinates
    aa_ppdb2 = PandasPdb()
    aa_ppdb2.df['ATOM'] = ppdb2.df['ATOM'][ppdb2.df['ATOM']['atom_name'] == 'CA']
    aa_ppdb2.df['ATOM'].loc[:, 'x_coord'] = all_atom_aligned[:, 0].tolist()
    aa_ppdb2.df['ATOM'].loc[:, 'y_coord'] = all_atom_aligned[:, 1].tolist()
    aa_ppdb2.df['ATOM'].loc[:, 'z_coord'] = all_atom_aligned[:, 2].tolist()

    # Construct output path
    aa_aligned_dest = "/".join(pdb2.split('/')[:-1]) + "/aa_aligned_" + pdb2.split('/')[-1]

    # Save aligned structure
    print(f"All-atom PDB destination {aa_aligned_dest}")
    aa_ppdb2.to_pdb(path=aa_aligned_dest, records=['ATOM'])

    return rmsd
