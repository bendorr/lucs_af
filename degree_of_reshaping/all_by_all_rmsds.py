"""
8.27.24
Ben Orr

all_by_all_rmsds.py

Calculate the pairwise RMSDs and TM scores between each pair of provided PDB structures.

Usage:
	all_by_all_rmsds.py --pdb_dirs <pdb_dir_paths>
		--outfile <output_file_path> 
		[ --pdb_dirs_subdir_split --pdb_suffix <pdb_suffix> 
		--num_tasks <num_tasks> --task_id <task_id> 
		--verbose --test_stop <int> ]

Inputs:

	pdb_dirs: Path or paths to directories containing pdb files named <pdb_name>.pdb,
			or containing subdirectories named <pdb_name>, in which case TM scores and RMSDs
			are calculated between each <pdb_dir>/<pdb_name>/<pdb_name>.pdb

	pdb_dirs_subdir_split: Specify that the provided pdb_dirs are split into subdirectories, named
			<pdb_name>. Calculate TM scores and RMSDs between each <pdb_dir>/<pdb_name>/<pdb_name>.pdb.

	outfile: Path to an output .csv file, containing pdb_names and their TM scores to the
			reference structure.

	pdb_suffix: Suffix at the end of <pdb_dir>/<pdb_name>/<pdb_name>_<pdb_suffix>.pdb files.

	num_tasks: Number of tasks by which to divide the calculations.
	
	task_id: ID of the current task (1-indexed, e.g. $SGE_TASK_ID for SGE job distributor).

	verbose: Print extra print statements.

	test_stop: Number of test calculations to perform.

Outputs:

	Numpy files with all-by-all RMSDs and TM scores, 
		as well as a json file with the order of PDBs in the matrices,

"""

import os
import argparse
import json
import numpy as np
import pandas as pd

# tmtools for calculating TM score
from tmtools.io import get_structure, get_residue_data
from tmtools.testing import get_pdb_path
from tmtools import tm_align


def calc_rmsd_and_tm_score(pdb_path_1, pdb_path_2):
	"""
	Use tmtools to calculate the RMSD and TM score between two PDB structures.
	"""
	tm_s1 = get_structure(pdb_path_1)
	tm_s2 = get_structure(pdb_path_2)
	tm_s1_chain = next(tm_s1.get_chains())
	tm_s2_chain = next(tm_s2.get_chains())
	return get_rmsd_TM_score(tm_s1_chain, tm_s2_chain)

def get_rmsd_TM_score(chain_1, chain_2):
	"""
	Use tmtools to calculate and return the RMSD and TM score between two tmtools chains.
	"""
	coords_1, seq_1 = get_residue_data(chain_1)
	coords_2, seq_2 = get_residue_data(chain_2)
	res = tm_align(coords_1, coords_2, seq_1, seq_2)

	return res.rmsd, res.tm_norm_chain1, res.tm_norm_chain2


if __name__ == "__main__":

	# Parse command line arguments.
	parser = argparse.ArgumentParser(
		description='Calculate the TM scores between each \
		provided PDB structure and a reference structure.',
		formatter_class=argparse.RawDescriptionHelpFormatter)

	parser.add_argument('--pdb_dirs', nargs='*', type=str, required=True,
			help='Path or paths to directories containing pdb files named <pdb_name>.pdb, \
			or containing subdirectories named <pdb_name>, in which case TM scores and RMSDs \
			are calculated between each <pdb_dir>/<pdb_name>/<pdb_name>.pdb')
	parser.add_argument('--pdb_dirs_subdir_split', action='store_true',
			help='Specify that the provided pdb_dirs are split into subdirectories, named \
			<pdb_name>. Calculate TM scores and RMSDs between each <pdb_dir>/<pdb_name>/<pdb_name>.pdb.')
	parser.add_argument('--outdir', type=str, required=True,
			help='Path to an output directory.')
	parser.add_argument('--pdb_suffix', type=str, required=False, default='',
			help='Suffix at the end of <pdb_dir>/<pdb_name>/<pdb_name>_<pdb_suffix>.pdb files. '
			'Currently not used')

	# Parallelization arguments
	parser.add_argument('--num_tasks', type=int, required=False, default=1,
			help='Number of tasks by which to divide the calculations.')
	parser.add_argument('--task_id', type=int, required=False, default=1,
			help='ID of the current task (1-indexed, e.g. $SGE_TASK_ID for SGE job distributor).')

	# Debugging arguments
	parser.add_argument('--verbose', action='store_true',
			help='Print extra print statements.')
	parser.add_argument('--test_stop', type=int, required=False, default=None,
			help='Number of test calculations to perform.')

	args = parser.parse_args()

	if args.test_stop != None:
		test_stop = 0

	# Load lists of PDB paths and names
	pdb_list = []
	pdb_names = []
	for pdb_dir in args.pdb_dirs:
		if args.pdb_dirs_subdir_split:
			for subdir in os.listdir(pdb_dir):
				for pdb in os.listdir(os.path.join(pdb_dir, subdir)):
					if pdb.endswith('.pdb'):
						pdb_list.append(os.path.join(pdb_dir, subdir, pdb))
						pdb_names.append(pdb.split('.')[0])
						continue # only store one PDB file per subdir
		else:
			for pdb in os.listdir(pdb_dir):
				if pdb.endswith('.pdb'):
					pdb_list.append(os.path.join(pdb_dir, pdb))
					pdb_names.append(pdb.split('.')[0])

	# Initialize matrices of RMSDs and TM scores
	rmsd_mat = np.zeros((len(pdb_list), len(pdb_list)))
	tm1_mat = np.zeros((len(pdb_list), len(pdb_list)))
	tm2_mat = np.zeros((len(pdb_list), len(pdb_list)))

	task_idx = 0

	for i in range(len(pdb_list)):

		for j in range(i+1, len(pdb_list)):

			task_idx += 1
			if task_idx % args.num_tasks != (args.task_id-1):
				continue

			pdb1 = pdb_list[i]
			pdb2 = pdb_list[j]

			rmsd, tm1, tm2 = calc_rmsd_and_tm_score(pdb1, pdb2)
			
			if args.verbose:
				print(f'Task idx: {task_idx}\nPDB1: {pdb1}\nPDB2: {pdb2}\nRMSD: {rmsd}\nTM1: {tm1}\nTM2: {tm2}\n')

			# Store the RMSD and TM scores in the matrices
			rmsd_mat[i][j] = rmsd
			tm1_mat[i][j] = tm1
			tm2_mat[i][j] = tm2
			
			if args.test_stop != None:
				if task_idx >= args.test_stop:
					break

		if args.test_stop != None:
			if task_idx >= args.test_stop:
				break

	if args.num_tasks == 1:
		outfile_suffix = ''
	else:
		outfile_suffix = f'_{args.task_id}'

	os.makedirs(args.outdir, exist_ok=True)
	np.save(os.path.join(args.outdir, f'rmsd{outfile_suffix}.npy'), rmsd_mat)
	np.save(os.path.join(args.outdir, f'tm1{outfile_suffix}.npy'), tm1_mat)
	np.save(os.path.join(args.outdir, f'tm2{outfile_suffix}.npy'), tm2_mat)
	with open(os.path.join(args.outdir, f'pdb_order{outfile_suffix}.json'), 'w') as f:
		json.dump(pdb_names, f)
	print(f'Saved outputs to: {args.outdir}')














