"""
Ben Orr
6.21.23

combine_matrices.py

Combine incomplete all-by-all (RMSD or TM score, for example) .npy matrices.

Usage:

	combine_helix_rmsd_matrices.py --data_folder <data_folder> --file_prefixes <file_prefix1> [ <file_prefix2> ... ] --fill_in

Inputs:

	data_folder: Folder containing .npy files with incomplete all-by-all matrices (all-by-all helix RMSDs, for example).

	file_prefixes: List of prefixes of the .npy files to be combined.

	fill_in: Only combine a preexisting {file_prefix}_full.npy matrix with a {file_prefix}_fill_in_mat.npy matrix.

"""

import os
import argparse
import numpy as np

if __name__ == '__main__':

	# Parse command line arguments.
	parser = argparse.ArgumentParser(
		description='Combine incomplete all-by-all (RMSD or TM score, for example) .npy matrices.',
		formatter_class=argparse.RawDescriptionHelpFormatter)

	parser.add_argument('--data_folder', type=str, required=True,
			help="Folder containing .npy files with incomplete all-by-all matrices (all-by-all helix RMSDs, for example).")
	# parser.add_argument('--file_prefixes', type=str, required=True,
	# 		help='Prefix of the .npy files to be combined.')
	parser.add_argument('--file_prefixes', nargs='*', type=str, required=True,
			help='List of prefixes of the .npy files to be combined.')
	parser.add_argument('--fill_in', action='store_true',
			help='Only combine a preexisting <file_prefix>_full.npy matrix with a <file_prefix>_fill_in_mat.npy matrix.')

	args = parser.parse_args()

	for file_prefix in args.file_prefixes:

		# If only filling in a preexisting <file_prefix>_full.npy matrix with a <file_prefix>_fill_in_mat.npy matrix
		if args.fill_in:
			files = [f for f in os.listdir(args.data_folder) if file_prefix in f and ('_full' in f or '_fill_in_mat' in f)]

		# If combining many incomplete all-by-all matrices
		else:
			files = [f for f in os.listdir(args.data_folder) if file_prefix in f and '_full' not in f]

		# print(files)
		# print(len(files))

		for i in range(len(files)):
			f = files[i]
			print(i)
			mat = np.load(os.path.join(args.data_folder, f))
			if i == 0:
				curr_mat = mat
			else:
				curr_mat = curr_mat + mat

		print('Combined numpy matrix:')
		print(curr_mat)

		np.save(os.path.join(args.data_folder, f'{file_prefix}_full'), curr_mat)
		print('Saved file: %s'%os.path.join(args.data_folder, f'{file_prefix}_full.npy'))


