"""
Calculates TM-scores between protein structures and a reference structure.

Author: Ben Orr
Date: 7.31.24

This script calculates Template Modeling (TM) scores between a set of PDB structures
and a reference structure. TM-score is a metric for measuring structural similarity
between protein models, with values ranging from 0 to 1 (higher = more similar).

The script supports:
- Parallel processing via task distribution
- Multiple input directories
- Customizable PDB file naming conventions
- Progress tracking and verbose output

Reference:
Zhang, Y. and Skolnick, J. (2004). "Scoring function for automated assessment of
protein structure template quality". Proteins, 57: 702-710.

Usage
-----
Basic usage:
    python tm_score_to_ref.py --pdb_dirs <dir1> [<dir2> ...] --ref_path <ref.pdb> --outfile <output.csv>

With parallel processing:
    python tm_score_to_ref.py --pdb_dirs <dir> --ref_path <ref.pdb> --outfile <output.csv> \\
        --num_tasks 10 --task_id $SGE_TASK_ID

Parameters
----------
pdb_dirs : List[str]
    Path or paths to directories containing subdirectories, each named <pdb_name>.
    TM scores are calculated between each <pdb_dir>/<pdb_name>/<pdb_name>_<pdb_suffix>.pdb
    structure and the reference PDB structure.

ref_path : str
    Path to a reference PDB file, to which TM scores will be calculated for all
    input PDB structures.

outfile : str
    Path to an output .csv file that will contain pdb_names and their TM scores
    to the reference structure.

pdb_suffix : str, optional
    Suffix at the end of <pdb_dir>/<pdb_name>/<pdb_name>_<pdb_suffix>.pdb files.
    Default is empty string (files named <pdb_name>.pdb).

num_tasks : int, optional
    Number of tasks by which to divide the calculations for parallel processing.
    Default is 1 (no parallelization).

task_id : int, optional
    ID of the current task (1-indexed). Use with job array systems like SGE's
    $SGE_TASK_ID variable. Default is 1.

verbose : bool, optional
    If set, print detailed progress information for each structure.

test_stop : int, optional
    Number of test calculations to perform before stopping. Useful for testing.

Output
------
CSV file containing columns:
    - name: PDB structure name
    - tm_score: TM score to reference structure (0-1 scale)

If num_tasks > 1, output file will be named <outfile>_<task_id>.csv
"""

import os
import argparse
from typing import Tuple, Any, List, Optional
import numpy as np
import pandas as pd

# tmtools for calculating TM score
from tmtools.io import get_structure, get_residue_data
from tmtools.testing import get_pdb_path
from tmtools import tm_align


def calc_TM_score(pdb_path_1: str, pdb_path_2: str) -> float:
    """
    Calculate TM score between two PDB structures.

    This function loads two PDB structures, extracts their first chains,
    and calculates the TM score between them using the tmtools library.

    Parameters
    ----------
    pdb_path_1 : str
        Path to the first PDB file.
    pdb_path_2 : str
        Path to the second PDB file (typically the reference structure).

    Returns
    -------
    float
        TM score normalized by the length of chain 1 (range: 0-1).
        Values > 0.5 generally indicate similar folds.

    Notes
    -----
    TM-score is length-independent and more sensitive to global topology
    than RMSD. A TM-score > 0.5 indicates the structures share the same fold.

    Examples
    --------
    >>> tm_score = calc_TM_score("model.pdb", "reference.pdb")
    >>> print(f"TM-score: {tm_score:.3f}")
    TM-score: 0.847
    """
    # Load both PDB structures
    tm_s1 = get_structure(pdb_path_1)
    tm_s2 = get_structure(pdb_path_2)

    # Extract first chain from each structure
    tm_s1_chain = next(tm_s1.get_chains())
    tm_s2_chain = next(tm_s2.get_chains())

    # Calculate and return TM score
    return get_TM_score(tm_s1_chain, tm_s2_chain)


def get_TM_score(chain_1: Any, chain_2: Any) -> float:
    """
    Calculate TM score between two tmtools chain objects.

    This is a lower-level function that operates on tmtools chain objects
    rather than file paths. It performs structural alignment and computes
    the TM score.

    Parameters
    ----------
    chain_1 : tmtools.Chain
        First protein chain object.
    chain_2 : tmtools.Chain
        Second protein chain object (typically the reference).

    Returns
    -------
    float
        TM score normalized by the length of chain 1 (range: 0-1).

    Notes
    -----
    The TM-score is asymmetric (tm_align provides both tm_norm_chain1 and
    tm_norm_chain2). This function returns the score normalized by chain 1's length.
    """
    # Extract coordinates and sequences from both chains
    coords_1, seq_1 = get_residue_data(chain_1)
    coords_2, seq_2 = get_residue_data(chain_2)

    # Perform TM alignment
    res = tm_align(coords_1, coords_2, seq_1, seq_2)

    # Return TM score normalized by chain 1 length
    return res.tm_norm_chain1


def calculate_tm_scores_to_ref(
    pdb_dirs: List[str],
    ref_path: str,
    pdb_suffix: str,
    num_tasks: int,
    task_id: int,
    verbose: bool = False,
    test_stop: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate TM scores between PDB structures and a reference structure.

    Parameters
    ----------
    pdb_dirs : List[str]
        Directories containing PDB subdirectories.
    ref_path : str
        Path to reference structure.
    pdb_suffix : str
        Suffix for PDB filenames.
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
    pd.DataFrame
        DataFrame with 'name' and 'tm_score' columns.
    """
    task_idx = 0
    results = {'name': [], 'tm_score': []}
    structures_processed = 0

    # Iterate through all PDB directories
    for pdb_dir in pdb_dirs:
        for subdir in os.listdir(pdb_dir):
            task_idx += 1

            # Skip if not assigned to current task
            if task_idx % num_tasks != (task_id - 1):
                continue

            # Construct PDB file path
            pdb_path = os.path.join(pdb_dir, subdir, f'{subdir}{pdb_suffix}.pdb')

            # Skip if file doesn't exist
            if not os.path.exists(pdb_path):
                print(f'File not found: {pdb_path}. Skipping this structure.')
                continue

            # Calculate TM score
            tm_score = calc_TM_score(pdb_path, ref_path)

            # Store results
            results['name'].append(subdir)
            results['tm_score'].append(tm_score)
            structures_processed += 1

            # Print progress if verbose
            if verbose:
                print(f'PDB path: {pdb_path}\nName: {subdir}\nTM Score: {tm_score:.4f}\n')

            # Stop early if in test mode
            if test_stop is not None and structures_processed >= test_stop:
                break

        if test_stop is not None and structures_processed >= test_stop:
            break

    return pd.DataFrame(results)


def save_tm_scores(df: pd.DataFrame, outfile: str, num_tasks: int, task_id: int) -> None:
    """
    Save TM scores DataFrame to CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing TM scores.
    outfile : str
        Output file path.
    num_tasks : int
        Total number of parallel tasks.
    task_id : int
        Current task ID.
    """
    # Determine output filename (add task ID suffix if running in parallel)
    if num_tasks == 1:
        output_path = outfile
    else:
        output_path = outfile.replace('.csv', f'_{task_id}.csv')

    # Save results
    df.to_csv(output_path, index=False)
    print(f'Saved {len(df)} TM scores to: {output_path}')


def main():
    """Main entry point for calculating TM scores to reference."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Calculate the TM scores between each provided PDB structure and a reference structure.',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Input/output arguments
    parser.add_argument(
        '--pdb_dirs',
        nargs='*',
        type=str,
        required=True,
        help='Path or paths to directories containing subdirectories, named <pdb_name>. '
             'TM scores are calculated between each <pdb_dir>/<pdb_name>/<pdb_name>.pdb structure '
             'and the reference PDB structure.')

    parser.add_argument(
        '--ref_path',
        type=str,
        required=True,
        help='Path to a reference PDB file, to which TM scores will be calculated for the PDB structures.')

    parser.add_argument(
        '--outfile',
        type=str,
        required=True,
        help='Path to an output .csv file, containing pdb_names and their TM scores to the reference structure.')

    parser.add_argument(
        '--pdb_suffix',
        type=str,
        required=False,
        default='',
        help='Suffix at the end of <pdb_dir>/<pdb_name>/<pdb_name>_<pdb_suffix>.pdb files.')

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

    # Calculate TM scores
    df = calculate_tm_scores_to_ref(
        args.pdb_dirs,
        args.ref_path,
        args.pdb_suffix,
        args.num_tasks,
        args.task_id,
        args.verbose,
        args.test_stop
    )

    # Save results
    save_tm_scores(df, args.outfile, args.num_tasks, args.task_id)


if __name__ == "__main__":
    main()


