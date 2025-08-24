"""
7.31.24
Ben Orr

tm_score_to_ref.py

Calculate the TM scores between each provided PDB structure and a reference structure.

Usage:
	tm_score_to_ref.py --pdb_dirs <pdb_dir_paths> --ref_path <ref_pdb_path> 
		--outfile <output_file_path> 
		[ --pdb_suffix <pdb_suffix> --num_tasks <num_tasks> 
		--task_id <task_id> --verbose --test_stop <int> ]

Inputs:

	pdb_dirs: Path or paths to directories containing subdirectories, named <pdb_name>. TM scores
		are calculated between each <pdb_dir>/<pdb_name>/<pdb_name>_<pdb_suffix>.pdb structure and 
		the reference PDB structure.

	ref_path: Path to a reference PDB file, to which TM scores will be calculated for the
		PDB structures.

	outfile: Path to an output .csv file, containing pdb_names and their TM scores to the
		reference structure.

	pdb_suffix: Suffix at the end of <pdb_dir>/<pdb_name>/<pdb_name>_<pdb_suffix>.pdb files.

	num_tasks: Number of tasks by which to divide the calculations.
	
	task_id: ID of the current task (1-indexed, e.g. $SGE_TASK_ID for SGE job distributor).

	verbose: Print extra print statements.

	test_stop: Number of test calculations to perform.

Outputs:

	Dataframe (.csv file) containing pdb_names and their TM scores to the
		reference structure.

"""

import os
import argparse
import numpy as np
import pandas as pd

# tmtools for calculating TM score
from tmtools.io import get_structure, get_residue_data
from tmtools.testing import get_pdb_path
from tmtools import tm_align


def calc_TM_score(pdb_path_1, pdb_path_2):
	"""
	Use tmtools to calculate the TM score between two PDB structures.
	"""
	tm_s1 = get_structure(pdb_path_1)
	tm_s2 = get_structure(pdb_path_2)
	tm_s1_chain = next(tm_s1.get_chains())
	tm_s2_chain = next(tm_s2.get_chains())
	return get_TM_score(tm_s1_chain, tm_s2_chain)

def get_TM_score(chain_1, chain_2):
	"""
	Use tmtools to calculate and return the TM score between two tmtools chains.
	"""
	coords_1, seq_1 = get_residue_data(chain_1)
	coords_2, seq_2 = get_residue_data(chain_2)
	res = tm_align(coords_1, coords_2, seq_1, seq_2)

	return res.tm_norm_chain1


if __name__ == "__main__":

	# Parse command line arguments.
	parser = argparse.ArgumentParser(
		description='Calculate the TM scores between each \
		provided PDB structure and a reference structure.',
		formatter_class=argparse.RawDescriptionHelpFormatter)

	parser.add_argument('--pdb_dirs', nargs='*', type=str, required=True,
			help='Path or paths to directories containing subdirectories, named <pdb_name>. \
			TM scores are calculated between each <pdb_dir>/<pdb_name>/<pdb_name>.pdb structure \
			and the reference PDB structure.')
	parser.add_argument('--ref_path', type=str, required=True,
			help='Path to a reference PDB file, to which TM scores will be calculated for the \
			PDB structures.')
	parser.add_argument('--outfile', type=str, required=True,
			help='Path to an output .csv file, containing pdb_names and their TM scores to the \
			reference structure.')
	parser.add_argument('--pdb_suffix', type=str, required=False, default='',
			help='Suffix at the end of <pdb_dir>/<pdb_name>/<pdb_name>_<pdb_suffix>.pdb files.')

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

	task_idx = 0

	d = {'name':[], 'tm_score':[]}
	for pdb_dir in args.pdb_dirs:
		for subdir in os.listdir(pdb_dir):

			task_idx += 1
			if task_idx % args.num_tasks != (args.task_id-1):
				continue
			
			pdb_path = os.path.join(pdb_dir, subdir, f'{subdir}{args.pdb_suffix}.pdb')
			if not os.path.exists(pdb_path):
				print(f'File not found: {pdb_path}. Skipping this structure.')
				continue
			tm_score = calc_TM_score(pdb_path, args.ref_path)
			d['name'].append(subdir)
			d['tm_score'].append(tm_score)
			
			if args.verbose:
				print(f'PDB path: {pdb_path}\nName: {subdir}\nTM Score: {tm_score}\n')
			
			if args.test_stop != None:
				test_stop += 1
				if test_stop == args.test_stop:
					break

	df = pd.DataFrame(d)
	if args.num_tasks == 1:
		outfile = args.outfile
	else:
		outfile = args.outfile.replace('.csv',f'_{args.task_id}.csv')
	df.to_csv(outfile)
	print(f'Saved file: {outfile}')


