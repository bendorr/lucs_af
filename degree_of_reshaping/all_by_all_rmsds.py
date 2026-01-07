"""
Computes pairwise RMSDs and TM-scores for all structure pairs in a dataset.

Author: Ben Orr
Date: 8.27.24

This script calculates pairwise RMSDs and TM-scores between all pairs of PDB structures
in a dataset, producing all-by-all similarity matrices. These matrices are useful for
clustering, outlier detection, and understanding structural diversity in protein datasets.

The script supports:
- Parallel processing via task distribution
- Flexible input directory structures (flat or hierarchical)
- Multiple similarity metrics (RMSD and TM-score)
- Both symmetric (RMSD) and asymmetric (TM-score) similarity measures

Output Matrices
---------------
Three matrices are generated:
1. RMSD matrix: Symmetric matrix of C-alpha RMSDs after optimal alignment
2. TM1 matrix: TM-scores normalized by structure 1 length
3. TM2 matrix: TM-scores normalized by structure 2 length

Note: Only the upper triangle of matrices is calculated (i < j) to avoid
redundant calculations, as RMSD is symmetric and TM-scores can be computed
in both directions simultaneously.

Usage
-----
Flat directory structure:
    python all_by_all_rmsds.py --pdb_dirs <dir1> [<dir2> ...] --outdir <output_dir>

Hierarchical directory structure:
    python all_by_all_rmsds.py --pdb_dirs <dir1> --pdb_dirs_subdir_split --outdir <output_dir>

With parallel processing:
    python all_by_all_rmsds.py --pdb_dirs <dir> --outdir <output_dir> \\
        --num_tasks 10 --task_id $SGE_TASK_ID

Parameters
----------
pdb_dirs : List[str]
    Path or paths to directories containing pdb files.
    - Flat structure: Contains .pdb files directly
    - Hierarchical structure: Contains subdirectories named <pdb_name>,
      each containing <pdb_name>.pdb or <pdb_name>_<suffix>.pdb

pdb_dirs_subdir_split : bool, optional
    If set, expect hierarchical directory structure where each pdb is in
    its own subdirectory named <pdb_name>.

outdir : str
    Path to output directory where matrices will be saved.

pdb_suffix : str, optional
    Suffix at the end of pdb files (e.g., "_relaxed" for <name>_relaxed.pdb).
    Currently not implemented.

num_tasks : int, optional
    Number of parallel tasks for distributing calculations. Default is 1.

task_id : int, optional
    ID of current task (1-indexed). Use with job array systems. Default is 1.

verbose : bool, optional
    If set, print detailed progress for each pairwise comparison.

test_stop : int, optional
    Number of pairwise comparisons to perform before stopping (for testing).

Output Files
------------
The following files are saved to outdir:
    - rmsd.npy or rmsd_<task_id>.npy: RMSD matrix (NxN, symmetric)
    - tm1.npy or tm1_<task_id>.npy: TM-score matrix normalized by structure 1
    - tm2.npy or tm2_<task_id>.npy: TM-score matrix normalized by structure 2
    - pdb_order.json or pdb_order_<task_id>.json: List of pdb names in matrix order

For parallel runs, matrices from different tasks can be summed element-wise
to produce the complete all-by-all matrix.
"""

import os
import argparse
import json
from typing import Tuple, List, Any
import numpy as np
import numpy.typing as npt
import pandas as pd

# tmtools for calculating TM score
from tmtools.io import get_structure, get_residue_data
from tmtools.testing import get_pdb_path
from tmtools import tm_align


def calc_rmsd_and_tm_score(pdb_path_1: str, pdb_path_2: str) -> Tuple[float, float, float]:
    """
    Calculate RMSD and TM-scores between two pdb structures.

    This function loads two pdb structures, extracts their first chains,
    performs optimal structural alignment, and calculates both RMSD and
    TM-scores between them.

    Parameters
    ----------
    pdb_path_1 : str
        Path to the first pdb file.
    pdb_path_2 : str
        Path to the second pdb file.

    Returns
    -------
    rmsd : float
        Root mean square deviation (Å) after optimal alignment.
    tm_norm_chain1 : float
        TM-score normalized by the length of chain 1 (0-1 scale).
    tm_norm_chain2 : float
        TM-score normalized by the length of chain 2 (0-1 scale).

    Notes
    -----
    TM-scores are asymmetric because they are normalized by structure length.
    For structures of similar length, tm_norm_chain1 ≈ tm_norm_chain2.
    RMSD is symmetric (RMSD(A,B) = RMSD(B,A)).

    Examples
    --------
    >>> rmsd, tm1, tm2 = calc_rmsd_and_tm_score("struct1.pdb", "struct2.pdb")
    >>> print(f"RMSD: {rmsd:.2f} Å, TM-score: {tm1:.3f}")
    RMSD: 3.45 Å, TM-score: 0.782
    """
    # Load both PDB structures
    tm_s1 = get_structure(pdb_path_1)
    tm_s2 = get_structure(pdb_path_2)

    # Extract first chain from each structure
    tm_s1_chain = next(tm_s1.get_chains())
    tm_s2_chain = next(tm_s2.get_chains())

    # Calculate and return RMSD and TM-scores
    return get_rmsd_TM_score(tm_s1_chain, tm_s2_chain)


def get_rmsd_TM_score(chain_1: Any, chain_2: Any) -> Tuple[float, float, float]:
    """
    Calculate RMSD and TM-scores between two tmtools chain objects.

    This is a lower-level function that operates on tmtools chain objects
    rather than file paths. It performs structural alignment and computes
    both RMSD and TM-scores.

    Parameters
    ----------
    chain_1 : tmtools.Chain
        First protein chain object.
    chain_2 : tmtools.Chain
        Second protein chain object.

    Returns
    -------
    rmsd : float
        Root mean square deviation (Å) after optimal alignment.
    tm_norm_chain1 : float
        TM-score normalized by the length of chain 1 (0-1 scale).
    tm_norm_chain2 : float
        TM-score normalized by the length of chain 2 (0-1 scale).

    Notes
    -----
    The tm_align function computes:
    - Optimal structural alignment using dynamic programming
    - RMSD over aligned residues
    - TM-scores normalized by both structure lengths

    The TM-score is more sensitive to global topology than RMSD and is
    length-independent, making it better for comparing structures of
    different sizes.
    """
    # Extract coordinates and sequences from both chains
    coords_1, seq_1 = get_residue_data(chain_1)
    coords_2, seq_2 = get_residue_data(chain_2)

    # Perform TM alignment (computes alignment, RMSD, and TM-scores)
    res = tm_align(coords_1, coords_2, seq_1, seq_2)

    # Return RMSD and both TM-score normalizations
    return res.rmsd, res.tm_norm_chain1, res.tm_norm_chain2


def collect_pdb_files(pdb_dirs: List[str], subdir_split: bool) -> Tuple[List[str], List[str]]:
    """
    Collect pdb files from input directories.

    Parameters
    ----------
    pdb_dirs : List[str]
        List of directories containing pdb files.
    subdir_split : bool
        If True, expect hierarchical directory structure.

    Returns
    -------
    Tuple[List[str], List[str]]
        List of pdb file paths and list of pdb names.
    """
    pdb_list: List[str] = []
    pdb_names: List[str] = []

    for pdb_dir in pdb_dirs:
        if subdir_split:
            # Hierarchical structure: <pdb_dir>/<pdb_name>/<pdb_name>.pdb
            for subdir in os.listdir(pdb_dir):
                subdir_path = os.path.join(pdb_dir, subdir)
                if not os.path.isdir(subdir_path):
                    continue
                for pdb in os.listdir(subdir_path):
                    if pdb.endswith('.pdb'):
                        pdb_list.append(os.path.join(subdir_path, pdb))
                        pdb_names.append(pdb.split('.')[0])
                        break  # Only store one pdb file per subdir
        else:
            # Flat structure: <pdb_dir>/<pdb_name>.pdb
            for pdb in os.listdir(pdb_dir):
                if pdb.endswith('.pdb'):
                    pdb_list.append(os.path.join(pdb_dir, pdb))
                    pdb_names.append(pdb.split('.')[0])

    return pdb_list, pdb_names


def calculate_pairwise_metrics(
    pdb_list: List[str],
    num_tasks: int,
    task_id: int,
    verbose: bool = False,
    test_stop: Optional[int] = None
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate all pairwise RMSDs and TM-scores.

    Parameters
    ----------
    pdb_list : List[str]
        List of pdb file paths.
    num_tasks : int
        Number of parallel tasks.
    task_id : int
        Current task ID (1-indexed).
    verbose : bool, optional
        Print detailed progress.
    test_stop : Optional[int], optional
        Stop after N calculations for testing.

    Returns
    -------
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
        RMSD matrix, TM1 matrix, and TM2 matrix.
    """
    # Initialize matrices (only upper triangle will be filled)
    n = len(pdb_list)
    rmsd_mat: npt.NDArray[np.float64] = np.zeros((n, n))
    tm1_mat: npt.NDArray[np.float64] = np.zeros((n, n))
    tm2_mat: npt.NDArray[np.float64] = np.zeros((n, n))

    task_idx = 0

    # Calculate pairwise metrics for upper triangle only
    for i in range(n):
        for j in range(i + 1, n):
            task_idx += 1

            # Skip if not assigned to current task
            if task_idx % num_tasks != (task_id - 1):
                continue

            pdb1 = pdb_list[i]
            pdb2 = pdb_list[j]

            # Calculate metrics
            rmsd, tm1, tm2 = calc_rmsd_and_tm_score(pdb1, pdb2)

            if verbose:
                print(f'Task idx: {task_idx}\nPDB1: {pdb1}\nPDB2: {pdb2}\n'
                      f'RMSD: {rmsd:.2f} Å\nTM1: {tm1:.3f}\nTM2: {tm2:.3f}\n')

            # Store results
            rmsd_mat[i][j] = rmsd
            tm1_mat[i][j] = tm1
            tm2_mat[i][j] = tm2

            # Stop early if in test mode
            if test_stop is not None and task_idx >= test_stop:
                return rmsd_mat, tm1_mat, tm2_mat

    return rmsd_mat, tm1_mat, tm2_mat


def save_results(
    outdir: str,
    rmsd_mat: npt.NDArray[np.float64],
    tm1_mat: npt.NDArray[np.float64],
    tm2_mat: npt.NDArray[np.float64],
    pdb_names: List[str],
    num_tasks: int,
    task_id: int
) -> None:
    """
    Save matrices and pdb order to files.

    Parameters
    ----------
    outdir : str
        Output directory path.
    rmsd_mat : npt.NDArray[np.float64]
        RMSD matrix.
    tm1_mat : npt.NDArray[np.float64]
        TM1 matrix.
    tm2_mat : npt.NDArray[np.float64]
        TM2 matrix.
    pdb_names : List[str]
        List of pdb names.
    num_tasks : int
        Total number of parallel tasks.
    task_id : int
        Current task ID.
    """
    # Determine output file suffix
    outfile_suffix = '' if num_tasks == 1 else f'_{task_id}'

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # Save matrices
    np.save(os.path.join(outdir, f'rmsd{outfile_suffix}.npy'), rmsd_mat)
    np.save(os.path.join(outdir, f'tm1{outfile_suffix}.npy'), tm1_mat)
    np.save(os.path.join(outdir, f'tm2{outfile_suffix}.npy'), tm2_mat)

    # Save PDB order
    with open(os.path.join(outdir, f'pdb_order{outfile_suffix}.json'), 'w') as f:
        json.dump(pdb_names, f, indent=2)

    # Print summary
    n = len(pdb_names)
    print(f'Saved outputs to: {outdir}')
    print(f'  - rmsd{outfile_suffix}.npy ({n}x{n} matrix)')
    print(f'  - tm1{outfile_suffix}.npy ({n}x{n} matrix)')
    print(f'  - tm2{outfile_suffix}.npy ({n}x{n} matrix)')
    print(f'  - pdb_order{outfile_suffix}.json ({n} structures)')


def main():
    """Main entry point for calculating pairwise RMSDs and TM-scores."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Calculate pairwise RMSDs and TM-scores for all pairs of pdb structures.',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Input arguments
    parser.add_argument(
        '--pdb_dirs',
        nargs='*',
        type=str,
        required=True,
        help='Path or paths to directories containing pdb files named <pdb_name>.pdb, '
             'or containing subdirectories named <pdb_name>, in which case TM scores and RMSDs '
             'are calculated between each <pdb_dir>/<pdb_name>/<pdb_name>.pdb')

    parser.add_argument(
        '--pdb_dirs_subdir_split',
        action='store_true',
        help='Specify that the provided pdb_dirs are split into subdirectories, named '
             '<pdb_name>. Calculate TM scores and RMSDs between each <pdb_dir>/<pdb_name>/<pdb_name>.pdb.')

    parser.add_argument(
        '--outdir',
        type=str,
        required=True,
        help='Path to an output directory.')

    parser.add_argument(
        '--pdb_suffix',
        type=str,
        required=False,
        default='',
        help='Suffix at the end of <pdb_dir>/<pdb_name>/<pdb_name>_<pdb_suffix>.pdb files. '
             'Currently not used')

    # Parallelization arguments
    parser.add_argument(
        '--num_tasks',
        type=int,
        required=False,
        default=1,
        help='Number of tasks by which to divide the calculations.')

    parser.add_argument(
        '--task_id',
        type=int,
        required=False,
        default=1,
        help='ID of the current task (1-indexed, e.g. $SGE_TASK_ID for SGE job distributor).')

    # Debugging arguments
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print extra print statements.')

    parser.add_argument(
        '--test_stop',
        type=int,
        required=False,
        default=None,
        help='Number of test calculations to perform.')

    args = parser.parse_args()

    # Collect pdb files
    pdb_list, pdb_names = collect_pdb_files(args.pdb_dirs, args.pdb_dirs_subdir_split)
    print(f'Found {len(pdb_list)} pdb structures')

    # Calculate pairwise metrics
    rmsd_mat, tm1_mat, tm2_mat = calculate_pairwise_metrics(
        pdb_list,
        args.num_tasks,
        args.task_id,
        args.verbose,
        args.test_stop
    )

    # Save results
    save_results(args.outdir, rmsd_mat, tm1_mat, tm2_mat, pdb_names, args.num_tasks, args.task_id)


if __name__ == "__main__":
    main()
