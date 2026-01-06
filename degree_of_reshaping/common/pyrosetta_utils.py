"""
PyRosetta utility functions for structural analysis.

Author: Ben Orr
Date: 12.18.25

This module consolidates PyRosetta helper functions used across the codebase,
providing utilities for:
- Residue numbering conversion (PDB ↔ Pose)
- Backbone coordinate extraction
- Pose alignment and superposition
- Residue selector string formatting

These functions require PyRosetta to be installed and initialized.

Key Functions
-------------
convert_pdb_to_pose_resnums : Convert PDB residue numbers to Pose numbering
convert_pose_to_pdb_resnums : Convert Pose residue numbers to PDB numbering
get_backbone_points : Extract backbone atom coordinates from pose
superimpose_poses : Superimpose one pose onto another
load_pyrosetta_strings : Format residue lists for PyRosetta selectors
get_residue_list_from_pdb : Extract all residue numbers from PDB file
"""

from typing import List, Tuple, Optional, Any
import numpy as np
import numpy.typing as npt

# PyRosetta imports - only used if PyRosetta is available
try:
    import pyrosetta
    from pyrosetta.rosetta.protocols.simple_moves import SuperimposeMover
    PYROSETTA_AVAILABLE = True
except ImportError:
    PYROSETTA_AVAILABLE = False


def convert_pdb_to_pose_resnums(
    pose: Any,
    pdb_res_list: List[int],
    chain: Optional[str] = None,
    verbose: bool = False
) -> List[int]:
    """
    Convert PDB residue numbers to PyRosetta pose numbering.

    Maps residue indices from PDB file numbering to PyRosetta's internal
    pose numbering system, which may differ due to chain breaks or non-standard
    numbering in PDB files.

    Parameters
    ----------
    pose : pyrosetta.Pose
        PyRosetta Pose object with PDB information.
    pdb_res_list : List[int]
        List of residue numbers in PDB numbering.
    chain : str, optional
        Chain ID to use for conversion. If None, uses chain of first residue.
    verbose : bool, optional
        If True, print conversion details and warnings. Default is False.

    Returns
    -------
    List[int]
        List of residue numbers in PyRosetta pose numbering.
        Residues that cannot be mapped (convert to 0) are excluded.

    Notes
    -----
    - Residues that cannot be mapped (convert to 0) are excluded from output
    - If chain is not provided, uses chain ID from the first residue
    - Prints warnings for excluded residues if verbose=True

    Examples
    --------
    >>> from pyrosetta import pose_from_file
    >>> pose = pose_from_file("structure.pdb")
    >>> pdb_nums = [10, 11, 12, 50, 51]
    >>> pose_nums = convert_pdb_to_pose_resnums(pose, pdb_nums, verbose=True)
    >>> print(pose_nums)
    [10, 11, 12, 48, 49]

    See Also
    --------
    convert_pose_to_pdb_resnums : Reverse conversion (Pose → PDB)
    """
    if not PYROSETTA_AVAILABLE:
        raise ImportError("PyRosetta is required for this function")

    # Determine chain ID
    if chain is None:
        chain = pose.pdb_info().pose2pdb(1).split()[1].strip()

    pose_res_list = []
    for pdb_res_num in pdb_res_list:
        # Convert PDB to Pose numbering
        pose_res_num = int(pose.pdb_info().pdb2pose(chain, pdb_res_num))

        if pose_res_num == 0:
            if verbose:
                print(f'PDB res {pdb_res_num} (chain {chain}) converted to {pose_res_num}. '
                      f'Excluding this residue.')
        else:
            pose_res_list.append(pose_res_num)

    if verbose:
        print(f'Converted PDB residue numbering to PyRosetta Pose numbers:')
        print(f'  PDB:  {pdb_res_list}')
        print(f'  Pose: {pose_res_list}')

    return pose_res_list


def convert_pose_to_pdb_resnums(
    pose: Any,
    pose_res_list: List[int],
    verbose: bool = False
) -> List[Tuple[int, str]]:
    """
    Convert PyRosetta pose numbering to PDB residue numbers.

    Parameters
    ----------
    pose : pyrosetta.Pose
        PyRosetta Pose object with PDB information.
    pose_res_list : List[int]
        List of residue numbers in Pose numbering.
    verbose : bool, optional
        If True, print conversion details. Default is False.

    Returns
    -------
    List[Tuple[int, str]]
        List of (residue_number, chain_id) tuples in PDB numbering.

    Examples
    --------
    >>> pose_nums = [10, 11, 12]
    >>> pdb_nums = convert_pose_to_pdb_resnums(pose, pose_nums)
    >>> print(pdb_nums)
    [(10, 'A'), (11, 'A'), (12, 'A')]

    See Also
    --------
    convert_pdb_to_pose_resnums : Reverse conversion (PDB → Pose)
    """
    if not PYROSETTA_AVAILABLE:
        raise ImportError("PyRosetta is required for this function")

    pdb_res_list = []
    for pose_res_num in pose_res_list:
        # Convert Pose to PDB: returns "ResNum ChainID"
        pdb_info_str = pose.pdb_info().pose2pdb(pose_res_num)
        res_num, chain_id = pdb_info_str.split()
        pdb_res_list.append((int(res_num), chain_id.strip()))

    if verbose:
        print(f'Converted Pose numbering to PDB residue numbering:')
        print(f'  Pose: {pose_res_list}')
        print(f'  PDB:  {pdb_res_list}')

    return pdb_res_list


def get_backbone_points(
    pose: Any,
    residues: List[int],
    atom_types: List[str] = ['N', 'CA', 'C']
) -> List[npt.NDArray[np.float64]]:
    """
    Extract backbone atom coordinates from a PyRosetta pose.

    Parameters
    ----------
    pose : pyrosetta.Pose
        PyRosetta pose object.
    residues : List[int]
        List of residue indices (1-indexed, Pose numbering) to extract.
    atom_types : List[str], optional
        List of atom names to extract. Default is ['N', 'CA', 'C'].
        Common options: ['CA'], ['N', 'CA', 'C'], ['N', 'CA', 'C', 'O'].

    Returns
    -------
    List[np.ndarray]
        List of coordinate arrays, one per atom. Each array has shape (3,).

    Notes
    -----
    Requires PyRosetta to be installed and initialized.
    Atoms are extracted in the order: residues × atom_types.

    Examples
    --------
    >>> from pyrosetta import pose_from_file
    >>> pose = pose_from_file("structure.pdb")
    >>> residues = [10, 11, 12, 13, 14]
    >>> # Extract only C-alpha atoms
    >>> ca_coords = get_backbone_points(pose, residues, atom_types=['CA'])
    >>> print(len(ca_coords))  # 5 atoms
    5
    >>> # Extract N, CA, C atoms
    >>> bb_coords = get_backbone_points(pose, residues, atom_types=['N', 'CA', 'C'])
    >>> print(len(bb_coords))  # 15 atoms (5 residues × 3 atoms)
    15

    See Also
    --------
    common.rmsd_calculations.calc_backbone_rmsd_pyrosetta : Uses this for RMSD
    """
    if not PYROSETTA_AVAILABLE:
        raise ImportError("PyRosetta is required for this function")

    points = []
    for residue in residues:
        for atom_type in atom_types:
            points.append(np.array(pose.residue(residue).xyz(atom_type)))
    return points


def superimpose_poses_by_residues(
    pose_source: Any,
    residues_source: List[int],
    pose_target: Any,
    residues_target: List[int],
    atom_types: List[str] = ['N', 'CA', 'C']
) -> None:
    """
    Superimpose source pose onto target pose using specified residues.

    Aligns pose_source onto pose_target by minimizing RMSD over the specified
    residue pairs. The pose_source is modified in place.

    Parameters
    ----------
    pose_source : pyrosetta.Pose
        Pose to be moved (modified in place).
    residues_source : List[int]
        Residue indices in source pose for alignment.
    pose_target : pyrosetta.Pose
        Reference pose (not modified).
    residues_target : List[int]
        Residue indices in target pose for alignment.
        Must have the same length as residues_source.
    atom_types : List[str], optional
        Atom types to use for superposition. Default is ['N', 'CA', 'C'].

    Notes
    -----
    - pose_source is modified in place by the superposition
    - Residue lists must have the same length
    - Uses PyRosetta's SuperimposeMover
    - For RMSD calculation after alignment, see rmsd_calculations module

    Examples
    --------
    >>> from pyrosetta import pose_from_file
    >>> design = pose_from_file("design.pdb")
    >>> reference = pose_from_file("reference.pdb")
    >>> # Align using helix residues (10-30)
    >>> helix_res = list(range(10, 31))
    >>> superimpose_poses_by_residues(design, helix_res, reference, helix_res)
    >>> # design is now aligned to reference

    See Also
    --------
    get_superimpose_transformation : Get transformation matrix without applying
    """
    if not PYROSETTA_AVAILABLE:
        raise ImportError("PyRosetta is required for this function")

    assert len(residues_source) == len(residues_target), \
        f"Residue lists must have same length: {len(residues_source)} vs {len(residues_target)}"

    # Create atom ID map for SuperimposeMover
    atom_map = pyrosetta.rosetta.std.map_core_id_AtomID_core_id_AtomID()

    for res_source, res_target in zip(residues_source, residues_target):
        for atom_type in atom_types:
            # Get atom IDs
            atom_id_source = pyrosetta.rosetta.core.id.AtomID(
                pose_source.residue(res_source).atom_index(atom_type), res_source
            )
            atom_id_target = pyrosetta.rosetta.core.id.AtomID(
                pose_target.residue(res_target).atom_index(atom_type), res_target
            )
            atom_map[atom_id_target] = atom_id_source

    # Perform superposition
    superimpose_mover = SuperimposeMover(atom_map)
    superimpose_mover.apply(pose_source)


def get_superimpose_transformation(
    pose_source: Any,
    residues_source: List[int],
    pose_target: Any,
    residues_target: List[int],
    atom_types: List[str] = ['N', 'CA', 'C']
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Get rotation matrix and translation vector for superposition.

    Calculates the optimal rotation and translation to align pose_source
    onto pose_target without modifying either pose.

    Parameters
    ----------
    pose_source : pyrosetta.Pose
        Source pose (not modified).
    residues_source : List[int]
        Residue indices in source pose.
    pose_target : pyrosetta.Pose
        Target pose (not modified).
    residues_target : List[int]
        Residue indices in target pose.
    atom_types : List[str], optional
        Atom types for alignment. Default is ['N', 'CA', 'C'].

    Returns
    -------
    rotation_matrix : np.ndarray
        3×3 rotation matrix.
    translation_vector : np.ndarray
        Translation vector of shape (3,).

    Examples
    --------
    >>> R, t = get_superimpose_transformation(pose1, res1, pose2, res2)
    >>> # Apply transformation manually
    >>> coords_aligned = coords @ R.T + t

    See Also
    --------
    superimpose_poses_by_residues : Apply superposition in place
    common.structural_alignment.kabsch : Alternative alignment method
    """
    if not PYROSETTA_AVAILABLE:
        raise ImportError("PyRosetta is required for this function")

    # Extract coordinates
    points_source = np.array(get_backbone_points(pose_source, residues_source, atom_types))
    points_target = np.array(get_backbone_points(pose_target, residues_target, atom_types))

    # Calculate centroids
    centroid_source = np.mean(points_source, axis=0)
    centroid_target = np.mean(points_target, axis=0)

    # Center coordinates
    centered_source = points_source - centroid_source
    centered_target = points_target - centroid_target

    # Calculate rotation using SVD
    H = centered_source.T @ centered_target
    U, S, Vt = np.linalg.svd(H)
    rotation_matrix = Vt.T @ U.T

    # Calculate translation
    translation_vector = centroid_target - centroid_source @ rotation_matrix.T

    return rotation_matrix, translation_vector


def load_pyrosetta_strings(
    seq_len: int,
    lhl_residues_list: List[List[int]]
) -> Tuple[List[str], List[str]]:
    """
    Format residue lists as PyRosetta selector strings.

    Creates residue selector strings for use with PyRosetta ResidueIndexSelector.
    Generates strings for both LHL (Loop-Helix-Loop) regions and non-LHL regions.

    Parameters
    ----------
    seq_len : int
        Total sequence length.
    lhl_residues_list : List[List[int]]
        List of LHL residue lists. Each inner list contains residue indices
        for one LHL unit.

    Returns
    -------
    non_lhl_strings : List[str]
        List of residue selector strings for non-LHL regions.
    lhl_strings : List[str]
        List of residue selector strings for LHL regions.

    Notes
    -----
    - Output strings are in PyRosetta format: "1-10,20-30"
    - Useful for creating ResidueIndexSelector objects
    - Non-LHL regions are calculated as gaps between LHL regions

    Examples
    --------
    >>> # Two LHL units in a 100-residue protein
    >>> lhl_list = [[10, 11, 12, 13, 14], [50, 51, 52, 53, 54]]
    >>> non_lhl, lhl = load_pyrosetta_strings(100, lhl_list)
    >>> print(non_lhl)
    ['1-9', '15-49', '55-100']
    >>> print(lhl)
    ['10-14', '50-54']

    See Also
    --------
    convert_pdb_to_pose_resnums : Convert PDB to Pose numbering first
    """
    # Flatten all LHL residues
    all_lhl_residues = []
    for lhl_res in lhl_residues_list:
        all_lhl_residues.extend(lhl_res)
    all_lhl_residues = sorted(set(all_lhl_residues))

    # Find non-LHL residues (gaps)
    all_residues = set(range(1, seq_len + 1))
    non_lhl_residues = sorted(all_residues - set(all_lhl_residues))

    def _residues_to_string(residues: List[int]) -> str:
        """Convert residue list to PyRosetta selector string."""
        if not residues:
            return ""

        ranges = []
        start = residues[0]
        end = residues[0]

        for i in range(1, len(residues)):
            if residues[i] == end + 1:
                end = residues[i]
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = residues[i]
                end = residues[i]

        # Add last range
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")

        return ",".join(ranges)

    # Create strings for non-LHL regions
    non_lhl_strings = []
    if non_lhl_residues:
        non_lhl_strings.append(_residues_to_string(non_lhl_residues))

    # Create strings for each LHL region
    lhl_strings = []
    for lhl_res in lhl_residues_list:
        lhl_strings.append(_residues_to_string(sorted(lhl_res)))

    return non_lhl_strings, lhl_strings


def get_residue_list_from_pdb(pdb_path: str) -> List[int]:
    """
    Extract all residue numbers from a PDB file.

    Parses ATOM records to find all unique residue numbers present in the
    PDB file. Useful for determining the residue range of a structure.

    Parameters
    ----------
    pdb_path : str
        Path to PDB file.

    Returns
    -------
    List[int]
        Sorted list of unique residue numbers found in the PDB file.

    Notes
    -----
    - Returns PDB residue numbers (not Pose numbers)
    - Only parses ATOM records (not HETATM)
    - Handles multi-chain PDBs

    Examples
    --------
    >>> residues = get_residue_list_from_pdb("structure.pdb")
    >>> print(f"Structure has {len(residues)} residues")
    >>> print(f"Residue range: {min(residues)}-{max(residues)}")
    Structure has 150 residues
    Residue range: 1-150
    """
    residues = set()

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                # PDB format: residue number is in columns 23-26 (1-indexed)
                res_num = int(line[22:26].strip())
                residues.add(res_num)

    return sorted(residues)
