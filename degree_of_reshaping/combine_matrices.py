"""
Combines incomplete all-by-all matrices into single complete matrix.

Author: Ben Orr
Date: 6.21.23

Combines incomplete all-by-all matrices (RMSD or TM-score) stored in separate .npy files
into single complete matrix. Useful when calculations distributed across multiple jobs.

Two modes:
1. Combine many partial matrices into full matrix
2. Fill missing values in existing full matrix

Usage
-----
Basic combination:
    python combine_matrices.py --data_folder <folder> --file_prefixes <prefix1> [<prefix2> ...]

Fill-in mode:
    python combine_matrices.py --data_folder <folder> --file_prefixes <prefix1> --fill_in

Parameters
----------
data_folder : str
    Folder with incomplete all-by-all .npy matrices, consistent prefixes

file_prefixes : List[str]
    Prefixes of .npy files to combine (e.g., "rmsd_matrix" for rmsd_matrix_1.npy)

fill_in : bool, optional
    Only combine {file_prefix}_full.npy with {file_prefix}_fill_in_mat.npy

Output
------
Saves a combined matrix as {file_prefix}_full.npy in the data_folder.
"""

import os
import argparse
from typing import List
import numpy as np
import numpy.typing as npt

def get_files_to_combine(data_folder: str, file_prefix: str, fill_in: bool) -> List[str]:
    """
    Files to combine based on mode.

    Parameters
    ----------
    data_folder : str
        Folder with matrix files
    file_prefix : str
        Prefix of files to combine
    fill_in : bool
        Only combine full and fill-in matrices

    Returns
    -------
    List[str]
        Filenames to combine
    """
    if fill_in:
        # Fill-in mode: Only combine existing full matrix with fill-in matrix
        files = [
            f for f in os.listdir(data_folder)
            if file_prefix in f and ('_full' in f or '_fill_in_mat' in f)
        ]
    else:
        # Combination mode: Combine all partial matrices (excluding any existing full matrix)
        files = [
            f for f in os.listdir(data_folder)
            if file_prefix in f and '_full' not in f
        ]
    return files


def combine_matrices(data_folder: str, files: List[str]) -> npt.NDArray[np.float64]:
    """
    Load and sum all matrices.

    Parameters
    ----------
    data_folder : str
        Folder with matrix files
    files : List[str]
        Filenames to combine

    Returns
    -------
    npt.NDArray[np.float64]
        Combined matrix

    Notes
    -----
    Assumes matrices initialized with zeros for missing values
    """
    curr_mat: npt.NDArray[np.float64] = None

    for i, f in enumerate(files):
        print(f'Processing file {i + 1}/{len(files)}: {f}')

        # Load matrix from file
        mat = np.load(os.path.join(data_folder, f))

        # Initialize or accumulate
        if i == 0:
            curr_mat = mat
        else:
            curr_mat = curr_mat + mat

    return curr_mat


def main():
    """Combine matrices."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Combine incomplete all-by-all (RMSD or TM score, for example) .npy matrices.',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--data_folder',
        type=str,
        required=True,
        help="Folder containing .npy files with incomplete all-by-all matrices (all-by-all helix RMSDs, for example).")

    parser.add_argument(
        '--file_prefixes',
        nargs='*',
        type=str,
        required=True,
        help='List of prefixes of the .npy files to be combined.')

    parser.add_argument(
        '--fill_in',
        action='store_true',
        help='Only combine a preexisting <file_prefix>_full.npy matrix with a <file_prefix>_fill_in_mat.npy matrix.')

    args = parser.parse_args()

    # Process each file prefix separately
    for file_prefix in args.file_prefixes:
        # Get files to combine
        files = get_files_to_combine(args.data_folder, file_prefix, args.fill_in)

        # Combine matrices
        combined_mat = combine_matrices(args.data_folder, files)

        # Display the combined matrix
        print('\nCombined numpy matrix:')
        print(combined_mat)

        # Save the combined matrix
        output_path = os.path.join(args.data_folder, f'{file_prefix}_full')
        np.save(output_path, combined_mat)
        print(f'Saved file: {output_path}.npy')


if __name__ == '__main__':
    main()


