"""
Unified RMSD (Root Mean Square Deviation) calculation utilities.

Author: Ben Orr
Date: 12.18.25

This module consolidates RMSD calculation functions from across the codebase,
providing both BioPython coordinate-based RMSD and PyRosetta-based RMSD
calculations with flexible atom selection.

The module provides:
- Simple coordinate-based RMSD (no dependencies on structure libraries)
- PyRosetta backbone RMSD with flexible atom type selection
- PyRosetta RMSDMetric-based RMSD for pre-aligned poses
- Segment-based RMSD with sliding window comparisons

Usage Guidelines
----------------
- Use `calc_rmsd()` for simple numpy coordinate arrays (BioPython style)
- Use `calc_backbone_rmsd_pyrosetta()` for PyRosetta poses with specific residues
- Use `calculate_rmsd_pyrosetta()` for PyRosetta RMSDMetric (pre-aligned poses)
- Use `rmsd_between_segments()` for sliding window segment comparisons

Key Functions
-------------
calc_rmsd : Simple coordinate-based RMSD
calc_backbone_rmsd_pyrosetta : PyRosetta backbone RMSD
calculate_rmsd_pyrosetta : PyRosetta RMSDMetric-based RMSD
rmsd_between_segments : Sliding window segment RMSD
get_backbone_points : Extract backbone atom coordinates from PyRosetta pose
"""

from typing import List, Optional, Tuple, Any
import numpy as np
import numpy.typing as npt

# PyRosetta imports - only used if PyRosetta functions are called
try:
    import pyrosetta
    from pyrosetta.rosetta.core.simple_metrics.metrics import RMSDMetric
    from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
    PYROSETTA_AVAILABLE = True
except ImportError:
    PYROSETTA_AVAILABLE = False


def calc_rmsd(ref_coords: npt.NDArray[np.float64], coords: npt.NDArray[np.float64]) -> float:
    """
    Calculate RMSD between two sets of coordinates (BioPython style).

    Computes RMSD as the square root of the mean squared Euclidean distance
    between corresponding points in two coordinate arrays. No alignment is
    performed; coordinates are compared in their given positions.

    Parameters
    ----------
    ref_coords : np.ndarray
        Reference coordinates, shape (n_points, n_dims).
        Typically 3D coordinates: (n_points, 3).
    coords : np.ndarray
        Comparison coordinates, shape (n_points, n_dims).
        Must have the same shape as ref_coords.

    Returns
    -------
    float
        Root Mean Square Deviation between the two coordinate sets.

    Notes
    -----
    - Typically used with 3D coordinates (n_dims=3)
    - Arrays must have the same shape
    - No alignment is performed; points are compared in given order
    - This is a pure NumPy implementation with no dependencies on structure libraries

    Examples
    --------
    >>> ref = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    >>> coords = np.array([[0.1, 0.1, 0.1], [1.1, 1.1, 1.1], [2.1, 2.1, 2.1]])
    >>> rmsd = calc_rmsd(ref, coords)
    >>> print(f"RMSD: {rmsd:.3f}")
    RMSD: 0.173

    See Also
    --------
    calc_backbone_rmsd_pyrosetta : For PyRosetta pose RMSD calculations
    """
    total_disp = 0.0
    for idx in range(len(ref_coords)):
        disp = 0.0
        for dim in range(len(ref_coords[idx])):
            disp += (ref_coords[idx][dim] - coords[idx][dim])**2
        total_disp += disp
    total_disp = total_disp / len(ref_coords)
    total_disp = total_disp**(1/2)
    return total_disp


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
        List of residue indices (1-indexed) to extract coordinates from.
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

    Examples
    --------
    >>> from pyrosetta import pose_from_file
    >>> pose = pose_from_file("structure.pdb")
    >>> residues = [10, 11, 12, 13, 14]
    >>> coords = get_backbone_points(pose, residues, atom_types=['CA'])
    >>> print(len(coords))  # 5 residues
    5
    """
    if not PYROSETTA_AVAILABLE:
        raise ImportError("PyRosetta is required for this function")

    points = []
    for residue in residues:
        for atom_type in atom_types:
            points.append(np.array(pose.residue(residue).xyz(atom_type)))
    return points


def calc_backbone_rmsd_pyrosetta(
    pose1: Any,
    residues1: List[int],
    pose2: Any,
    residues2: List[int],
    atom_types: List[str] = ['N', 'CA', 'C']
) -> Optional[float]:
    """
    Calculate backbone RMSD between two PyRosetta poses for specific residues.

    Extracts backbone atom coordinates from specified residues in both poses
    and calculates RMSD between them. Useful for comparing specific regions
    (e.g., helices, loops) between two structures.

    Parameters
    ----------
    pose1 : pyrosetta.Pose
        First PyRosetta pose.
    residues1 : List[int]
        Residue indices (1-indexed) in first pose.
    pose2 : pyrosetta.Pose
        Second PyRosetta pose.
    residues2 : List[int]
        Residue indices (1-indexed) in second pose.
        Must have the same length as residues1.
    atom_types : List[str], optional
        Backbone atom types to include in RMSD calculation.
        Default is ['N', 'CA', 'C'] (heavy backbone atoms).
        Other options: ['CA'] (C-alpha only), ['N', 'CA', 'C', 'O'] (all backbone).

    Returns
    -------
    float or None
        Backbone RMSD value in Ångstroms.
        Returns None if RMSD calculation fails (e.g., missing residues).

    Notes
    -----
    - Residue lists must have the same length
    - No alignment is performed before RMSD calculation
    - If residues are missing, returns None with a warning message
    - Requires PyRosetta to be installed

    Examples
    --------
    >>> from pyrosetta import pose_from_file
    >>> pose1 = pose_from_file("design.pdb")
    >>> pose2 = pose_from_file("prediction.pdb")
    >>> # Compare helix region (residues 10-30)
    >>> helix_residues = list(range(10, 31))
    >>> rmsd = calc_backbone_rmsd_pyrosetta(
    ...     pose1, helix_residues, pose2, helix_residues, atom_types=['CA']
    ... )
    >>> print(f"Helix CA-RMSD: {rmsd:.2f} Å")
    Helix CA-RMSD: 1.23 Å

    See Also
    --------
    calc_rmsd : For simple coordinate-based RMSD
    calculate_rmsd_pyrosetta : For RMSDMetric-based calculations
    """
    if not PYROSETTA_AVAILABLE:
        raise ImportError("PyRosetta is required for this function")

    assert len(residues1) == len(residues2), \
        f"Residue lists must have same length: {len(residues1)} vs {len(residues2)}"

    def _rmsd(points1: List[npt.NDArray[np.float64]],
              points2: List[npt.NDArray[np.float64]]) -> Optional[float]:
        """
        Calculate RMSD between two lists of numpy point arrays.

        Parameters
        ----------
        points1 : List[np.ndarray]
            First set of 3D coordinate arrays.
        points2 : List[np.ndarray]
            Second set of 3D coordinate arrays.

        Returns
        -------
        float or None
            RMSD value, or None if calculation fails.
        """
        diff = [points1[i] - points2[i] for i in range(len(points1))]

        if len(diff) == 0:
            print('\n\nCurrent design is missing helix residues in its design model! '
                  'Setting helix RMSD = None for this region.\n\n')
            return None

        try:
            return np.sqrt(sum(np.dot(d, d) for d in diff) / len(diff))
        except Exception as e:
            print(f'\n\nError calculating RMSD: {e}\n'
                  'Setting RMSD = None for this region.\n\n')
            return None

    # Extract backbone coordinates from both poses
    points1 = get_backbone_points(pose1, residues1, atom_types=atom_types)
    points2 = get_backbone_points(pose2, residues2, atom_types=atom_types)

    return _rmsd(points1, points2)


def rmsd_between_segments(
    pose1: Any,
    segment1: Tuple[int, int],
    pose2: Any,
    segment2: Tuple[int, int],
    atom_types: List[str] = ['N', 'CA', 'C']
) -> float:
    """
    Calculate minimum backbone RMSD between two segments using sliding windows.

    Compares two segments of potentially different lengths by sliding the shorter
    segment along the longer one and finding the minimum RMSD. Useful for finding
    the best alignment between similar structural elements (e.g., helices).

    Parameters
    ----------
    pose1 : pyrosetta.Pose
        First PyRosetta pose.
    segment1 : Tuple[int, int]
        (start, stop) residue indices (1-indexed, inclusive) in pose1.
    pose2 : pyrosetta.Pose
        Second PyRosetta pose.
    segment2 : Tuple[int, int]
        (start, stop) residue indices (1-indexed, inclusive) in pose2.
    atom_types : List[str], optional
        Backbone atom types to include. Default is ['N', 'CA', 'C'].

    Returns
    -------
    float
        Minimum RMSD found across all sliding window positions.

    Notes
    -----
    - If segments have different lengths, the shorter segment is slid along the longer
    - Multiple RMSD values are calculated for different window positions
    - Returns the minimum RMSD across all positions
    - Requires PyRosetta to be installed

    Examples
    --------
    >>> # Compare two helices of different lengths
    >>> helix1 = (10, 25)  # 16 residues
    >>> helix2 = (15, 35)  # 21 residues
    >>> min_rmsd = rmsd_between_segments(pose1, helix1, pose2, helix2)
    >>> print(f"Best helix alignment RMSD: {min_rmsd:.2f} Å")
    Best helix alignment RMSD: 0.87 Å

    See Also
    --------
    calc_backbone_rmsd_pyrosetta : For fixed residue comparisons
    """
    if not PYROSETTA_AVAILABLE:
        raise ImportError("PyRosetta is required for this function")

    # Ensure segment2 is the longer segment (swap if needed)
    if segment2[1] - segment2[0] < segment1[1] - segment1[0]:
        pose1, segment1, pose2, segment2 = pose2, segment2, pose1, segment1

    length1 = segment1[1] - segment1[0] + 1
    length_diff = (segment2[1] - segment2[0]) - (segment1[1] - segment1[0])

    # Calculate RMSDs for different sliding window positions
    rmsds = []
    residues1 = list(range(segment1[0], segment1[1] + 1))

    for shift in range(length_diff + 1):
        residues2 = list(range(segment2[0] + shift, segment2[0] + length1 + shift))
        rmsd = calc_backbone_rmsd_pyrosetta(
            pose1, residues1, pose2, residues2, atom_types=atom_types
        )
        if rmsd is not None:
            rmsds.append(rmsd)

    return min(rmsds) if rmsds else float('inf')


def calculate_rmsd_pyrosetta(
    ref_pose: Any,
    mobile_pose: Any,
    rmsd_res_string_list: List[str],
    rmsd_type: Any = None
) -> float:
    """
    Calculate RMSD using PyRosetta's RMSDMetric for pre-aligned poses.

    Uses PyRosetta's RMSDMetric class to calculate RMSD over specified residue
    ranges. This function assumes poses are already aligned and does not perform
    superposition. Useful for calculating RMSD over multiple regions.

    Parameters
    ----------
    ref_pose : pyrosetta.Pose
        Reference pose (already aligned).
    mobile_pose : pyrosetta.Pose
        Mobile pose to compare (already aligned to ref_pose).
    rmsd_res_string_list : List[str]
        List of residue range strings in PyRosetta format.
        Examples: ["1-100"], ["10-50", "60-80"], ["1-50,60-100"].
    rmsd_type : pyrosetta.rosetta.core.scoring.rmsd_atoms.RmsdType, optional
        Rosetta RMSD atom type. Default is rmsd_protein_bb_heavy_including_O.
        Other options: rmsd_protein_bb_ca, rmsd_protein_bb_heavy, etc.

    Returns
    -------
    float
        Average RMSD across all specified residue ranges.

    Notes
    -----
    - Poses must be pre-aligned (no superposition is performed)
    - If poses have different lengths, automatically adjusts last range
    - Calculates RMSD for each range separately, then returns the mean
    - Requires PyRosetta to be installed and initialized

    Examples
    --------
    >>> from pyrosetta import pose_from_file
    >>> from pyrosetta.rosetta.core.scoring.rmsd_atoms import rmsd_protein_bb_ca
    >>> ref = pose_from_file("reference.pdb")
    >>> mobile = pose_from_file("model.pdb")
    >>> # Calculate CA-only RMSD for two regions
    >>> rmsd = calculate_rmsd_pyrosetta(
    ...     ref, mobile, ["10-30", "50-70"], rmsd_type=rmsd_protein_bb_ca
    ... )
    >>> print(f"Average region RMSD: {rmsd:.2f} Å")
    Average region RMSD: 1.45 Å

    See Also
    --------
    calc_backbone_rmsd_pyrosetta : For unaligned poses with specific residues
    """
    if not PYROSETTA_AVAILABLE:
        raise ImportError("PyRosetta is required for this function")

    # Set default RMSD type if not provided
    if rmsd_type is None:
        rmsd_type = pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_heavy_including_O

    # Handle length mismatches by adjusting the last residue range
    if len(ref_pose.sequence()) > len(mobile_pose.sequence()) and \
            str(len(ref_pose.sequence())) in rmsd_res_string_list[-1]:
        rmsd_res_string_list[-1] = \
            rmsd_res_string_list[-1].replace(
                f'-{len(ref_pose.sequence())}',
                f'-{len(mobile_pose.sequence())}'
            )

    rmsd_list = []

    # Calculate RMSD for each residue range
    for rmsd_res_string in rmsd_res_string_list:
        rmsd_res = ResidueIndexSelector(rmsd_res_string)

        # Set up the RMSDMetric
        rmsd_metric = RMSDMetric()
        rmsd_metric.set_run_superimpose(False)  # Assumes pre-aligned poses
        rmsd_metric.set_comparison_pose(ref_pose)
        rmsd_metric.set_rmsd_type(rmsd_type)

        # Calculate RMSD over the specified residues
        rmsd_metric.set_residue_selector(rmsd_res)
        rmsd_metric.set_residue_selector_reference(rmsd_res)
        rmsd = rmsd_metric.calculate(mobile_pose)

        rmsd_list.append(rmsd)

    return np.mean(rmsd_list)
