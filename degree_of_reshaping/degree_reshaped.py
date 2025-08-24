"""
degree_reshaped.py

Ben Orr
8.21.23

The script takes in LUCS designs (Rosetta models, and optionally AF2/3 predictions)
then calculates RMSD/structural metrics (including helix RMSDs to a reference structure)
and AF2/3 confidence metrics (pLDDT and pAE) for LUCS designs' reshaped residues.

Uses PyRosetta for alignment and RMSD calculations.

"""

import argparse
from Bio.PDB.DSSP import DSSP
import os
import sys
import json
import gzip
import shutil
import glob
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pickle as pkl
import pandas as pd

# For Reshaped Helix Angle and Displacement
from Bio.PDB.DSSP import DSSP
from Bio.PDB import *

# For conversion between CIF and PDB format
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBIO

# For helix RMSD calculations
import pyrosetta
from pyrosetta import rosetta

# For alignment and RMSD calculations
from pyrosetta import init
import pyrosetta.distributed.dask
from pyrosetta import pose_from_file
from pyrosetta.rosetta.core.simple_metrics.metrics import RMSDMetric
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector, OrResidueSelector
from pyrosetta.rosetta.protocols.fold_from_loops.movers import AlignByResidueSelectorMover
# init pyrosetta
init(options='-mute all')

# tmtools for calculating TM score
from tmtools.io import get_structure, get_residue_data
from tmtools.testing import get_pdb_path
from tmtools import tm_align

# Import utility functions for degree_reshaped.py
import utils_degree_reshaped as utils



if __name__ == "__main__":

	# Parse command line arguments.
	parser = argparse.ArgumentParser(
		description='Calculate Degree of Reshaping metrics on LUCS designs\
		and their AF2/3 predictions.',
		formatter_class=argparse.RawDescriptionHelpFormatter)

	# LUCS inputs
	parser.add_argument('--lucs_data', nargs='*', type=str, required=True,
			help='Path to a directory or directories containing LUCS designs.')
	parser.add_argument('--lucs_data_subdir_split', action='store_true',
			help='Indicate whether the LUCS designs are split into subdirectories of the\
			lucs_data directory.')
	parser.add_argument('--lucs_af_design_ids_match', action='store_true',
			help="Indicate whether the LUCS design IDs match the AF design IDs exactly. The alternative \
			is that the LUCS design ID is af_design_id.split('_')[0].")
	parser.add_argument('--original_lucs_data', type=str, required=False, default='',
			help='Path to a directory containing the backbones output by LUCS. This differs from the lucs_data \
			folder in that the lucs_data folder may contain Rosetta models of threaded design sequences onto Rosetta\
			backbones.')
	parser.add_argument('--original_lucs_data_subdir_split', action='store_true',
			help='Indicate whether the Original LUCS designs are split into subdirectories of the\
			lucs_data directory.')
	parser.add_argument('--original_LUCS_data_compatible_LHLs', action='store_true',
			help='Indicate whether the Original LUCS designs are contained in a \
			_screen_compatible_loop_helix_loop_units directory (whose filenames are model_<lucs_design_id>.pdb).')
	parser.add_argument('--starting_structure_path', type=str, required=False,
			default='/wynton/home/kortemme/borr/thesis/local_protein_sequence_design/test_inputs/2lv8_inputs/2lv8_cleaned.pdb',
			help='Path to the LUCS starting structure.')
	parser.add_argument('--starting_structure_insertion_points', type=str, required=False,
			default='/wynton/home/kortemme/borr/thesis/local_protein_sequence_design/test_inputs/2lv8_inputs/2lv8_insertion_points.json',
			help='Path to the LUCS starting structures insertion points file.')
	parser.add_argument('--start_struct_res_start_idx', type=int, required=False, default=1,
			help='Index on which the starting structures residue numbers begin.')
	parser.add_argument('--filter_by_subsequence', type=str, required=False, default='',
			help='If a designs sequence does not contain this subsequence, then skip the design.')
	parser.add_argument('--lucs_id_suffix', type=str, required=False, default='',
			help='Suffix added to <design_id>/<design_id><suffix>.pdb file names.')
	parser.add_argument('--include_designs_df', type=str, required=False, default=None,
			help='Path to a .csv file with a design_id column. If provided, only these design IDs '
			'will be analyzed.')

	# RMSD inputs
	parser.add_argument('--use_dssp', action='store_true',
			help='Use DSSP secondary structure calculations to identify helical residues\
			in the reshaped regions.')
	parser.add_argument('--assume_helix', action='store_true',
			help='If fewer than 4 helical residues are found with DSSP, then assume res_list[5:-5] is helical.')
	parser.add_argument('--bb_rem_all_helical', action='store_true',
			help="Assume that all designs' bb_remodeled_residues (in the design_info.json files) are helical.")
	parser.add_argument('--starting_structure_helix_residues_file', type=str, required=False, default=None,
			help='Use a provided json file containing (a list of lists of) helix residues in the Starting Structure. \
			If not provided, then use DSSP to find reshaped helix residues in the Starting Structure.')

	# Alignment inputs
	parser.add_argument('--use_align_residues_file', action='store_true',
			help='Align the design structures on the residues contained in their <lucs_dir>/<design_id>_align_residues.json \
			files, which contain lists of lists of residues.')
	parser.add_argument('--starting_structure_align_residues_file', type=str, required=False, default=None,
			help='If using --use_align_residues_file to select residues on which to align for each design, then \
			provide an align_residues.json file (list of lists of residue indices) for the starting structure.')
	parser.add_argument('--aligned_pdb_outdir', type=str, required=False, default=None,
			help='If provided, then save aligned PDB files when calculating Helix RMSD to the starting structure \
			to the specified directory.')

	# AlphaFold inputs
	parser.add_argument('--af_data_dirs', nargs='*', type=str, required=False, default=None,
			help='Path to a directory or directories containing AF predictions for LUCS designs.\
			If not provided, then skip analyis of AF predictions.')
	parser.add_argument('--af_data_parents', nargs='*', type=str, required=False, default=None,
			help='Path to a parent directory or parent directories containing subdirectories with AF predictions for LUCS designs.')
	parser.add_argument('--is_colabfold_format', action='store_true',
			help='Indicate whether the AF data dir is in Colabfold output format (all PDB and json\
			files for all designs in the same directory).')
	parser.add_argument('--use_relaxed_af', action='store_true',
			help='Use relaxed AF predictions.')
	parser.add_argument('--use_af_sequence', action='store_true',
			help='Store design sequences based off their AF predictions sequences.')

	# AlphaFold3 inputs
	parser.add_argument('--is_af3', action='store_true',
			help='Indicate whether the AlphaFold predictions are from AlphaFold3, and in the format output by the AlphaFold Server.')
	parser.add_argument('--is_af3_linux', action='store_true',
			help='Indicate whether the AlphaFold predictions are from AlphaFold3, and in the format output by AF3 linux installations.')
	parser.add_argument('--af_prefix', type=str, required=False, default='',
			help='Prefix in front of design_id in AF2/3 predictions, such as fold_ for AF3 server outputs.')
	parser.add_argument('--af_is_cif', action='store_true',
			help='Indicate that the AF predictions are .cif files.')
	
	# ProteinMPNN designs inputs
	parser.add_argument('--pmpnn_suffixes', nargs='*', type=str, required=False, default=[''],
			help='A list of suffixes for multiple ProteinMPNN sequences for a single LUCS backbone.')

	# Parallelization arguments
	parser.add_argument('--num_tasks', type=int, required=False, default=1,
			help='Number of tasks by which to divide the calculations.')
	parser.add_argument('--task_id', type=int, required=False, default=1,
			help='ID of the current task (1-indexed, e.g. $SGE_TASK_ID for SGE job distributor).')

	# Output arguments
	parser.add_argument('--outfolder', type=str, required=True,
			help='Path to the output folder for figures and files containing clustered data.')
	parser.add_argument('--outfile', type=str, required=True,
			help='Name for the output dataframe.')
	parser.add_argument('--skip_completed_df_dir', type=str, required=False, default=None,
			help='If provided, search for the df that should be written out by the current task. If this dataframe \
			exists, then finish the current task.')

	# Debugging arguments
	parser.add_argument('--verbose', action='store_true',
			help='Print extra print statements.')
	parser.add_argument('--test_stop', type=int, required=False, default=None,
			help='Number of test calculations to perform.')

	args = parser.parse_args()

	# Check if the current task's df has already been written out.
	# If it has, then end the current task.
	if args.skip_completed_df_dir != None:
		outfile = args.outfile.replace('.csv', f'_{args.task_id}.csv')
		if os.path.exists(os.path.join(args.skip_completed_df_dir, outfile)):
			print(f'DataFrame {os.path.join(args.skip_completed_df_dir, outfile)} already exists. Skipping this task.')
			exit()

	# Make output folder
	os.makedirs(args.outfolder, exist_ok=True)

	# Load the Starting Structure 
	start_struct = pose_from_file(args.starting_structure_path)

	# Select the non-backbone_remodeled residues in the Starting
	# Structure so they can be aligned to the non-backbone_remodeled
	# residues of the LUCS structures
	pdb_ss_bb_remodeled_residues, pdb_ss_non_bb_remodeled_residues, ss_start_stop_idxs = \
		utils.get_remodeled_residues_from_insertion_points(
			args.starting_structure_insertion_points, 
			args.start_struct_res_start_idx,
			len(start_struct.sequence()))

	if args.verbose:
		print('Starting Structure bb_remodeled_residues (PDB-numbered):')
		print(pdb_ss_bb_remodeled_residues)
		print()

	# Load a list of lists of Starting Structure LHL residues
	if args.starting_structure_helix_residues_file:
		pdb_ss_lhl_residues, pdb_ss_helix_residues = utils.load_ss_lhl_res_from_file(
															args.starting_structure_helix_residues_file, 
															pdb_ss_bb_remodeled_residues)
	else:
		pdb_ss_lhl_residues, pdb_ss_helix_residues = utils.load_ss_lhl_res(start_struct, pdb_ss_bb_remodeled_residues)

	if args.verbose:
		print('Starting Structure LHL residues:')
		for res_list in pdb_ss_lhl_residues:
			print(res_list)
		print('Starting Structure Helix residues:')
		for res_list in pdb_ss_helix_residues:
			print(res_list)

	### Convert between PDB and PyRosetta residue numberings for Starting Structure
	ss_bb_remodeled_residues = utils.convert_pdb_to_pose_resnums(start_struct, pdb_ss_bb_remodeled_residues, verbose=args.verbose)
	ss_non_bb_remodeled_residues = utils.convert_pdb_to_pose_resnums(start_struct, pdb_ss_non_bb_remodeled_residues, verbose=args.verbose)
	ss_lhl_residues = []
	for i in range(len(pdb_ss_lhl_residues)):
		ss_lhl_residues.append([])
		ss_lhl_residues[i] = utils.convert_pdb_to_pose_resnums(start_struct, pdb_ss_lhl_residues[i], verbose=args.verbose)
	ss_helix_residues = []
	for i in range(len(pdb_ss_helix_residues)):
		ss_helix_residues.append([])
		ss_helix_residues[i] = utils.convert_pdb_to_pose_resnums(start_struct, pdb_ss_helix_residues[i], verbose=args.verbose)

	# Load strings containing residue ranges for PyRosetta
	ss_non_bb_remodeled_residues_string_list, ss_bb_remodeled_residues_string_list = \
		utils.load_pyrosetta_strings(args.starting_structure_path, ss_lhl_residues)

	# Load a DataFrame with all the degree of reshaping and confidence data.
	# These data will be held in a list of dictionaries.
	sub_dicts = []

	# First, gather a list of design IDs and their AF folders from the AF output.
	design_ids_af_dirs = utils.get_design_ids_af_dirs(args)

	if args.verbose:
		print('Number of design IDs and their AF dirs:')
		print(len(design_ids_af_dirs))
		print()

	## The Main Loop:
	task_idx = 0 # I should simply use enumerate() below.

	# Iterate over the design IDs and their AF outputs dirs
	for lucs_design_id, af_design_id, af_dir in design_ids_af_dirs:

		task_idx += 1
		if task_idx % args.num_tasks != (args.task_id-1):
			continue
		
		if args.verbose:
			print(f'\n\n\nLUCS design ID: {lucs_design_id}')
			print(f'AF design ID: {af_design_id}')
			print(f'AF dir: {af_dir}')

		# Find the LUCS dir with the current design, if 
		# the LUCS data is split into subdirs
		
		# If the lucs_design_ids are just the prefix to the full
		# design_id (like in 100065_mpnnSeq1), then I want to
		# find the lucs_der with just the lucs_design_id
		# AF3 design IDs contain the appropriate AF3 prefix (--args.af_prefix 'fold_') # if args.is_af3 or args.lucs_af_design_ids_match ...?
		if args.lucs_af_design_ids_match:
			lucs_dir = utils.find_lucs_dir(args, af_design_id)
		else:
			lucs_dir = utils.find_lucs_dir(args, lucs_design_id)

		# sub_dict will become a row in the output df
		sub_dict = {}
		sub_dict['design_id'] = af_design_id
		sub_dict['af_design_id'] = af_design_id
		sub_dict['lucs_design_id'] = lucs_design_id
		sub_dict['af_location'] = af_dir
		sub_dict['lucs_location'] = lucs_dir

		# Find the original LUCS backbone's location
		if args.original_lucs_data != '':
			original_lucs_dir = utils.find_original_lucs_dir(args, lucs_design_id)
			sub_dict['original_lucs_location'] = utils.find_original_lucs_dir(args, lucs_design_id)
		else:
			sub_dict['original_lucs_location'] = 'Not Provided'

		# If the original LUCS backbone is provided, then use this structure as the FastDesign reference structure:
		if args.original_lucs_data != '':
			lucs_struct_dir = original_lucs_dir
		else:
			lucs_struct_dir = lucs_dir

		# If the design structure has not been unzipped, then unzip the design.pdb.gz
		# file and copy it to a new file, unzipped_design.pdb


		### TO DO: Add a flag that specifies which lucs_struct_path to use.


		# Search for multiple possible LUCS structure filenames (some LUCS structures are
		# named unzipped_design.pdb, while others are named {af_design_id}.pdb)
		lucs_struct_filenames = [f'{af_design_id}{args.lucs_id_suffix}.pdb', 'unzipped_design.pdb']
		lucs_struct_paths = []
		for lucs_struct_file in lucs_struct_filenames:
			if args.verbose:
				print(f'Looking for LUCS file: {os.path.join(lucs_struct_dir, lucs_struct_file)}')
				print(f'Found LUCS Files: {glob.glob(os.path.join(lucs_struct_dir, lucs_struct_file))}')
			for found_file in glob.glob(os.path.join(lucs_struct_dir, lucs_struct_file)):
				lucs_struct_paths.append(found_file)
		lucs_struct_path = lucs_struct_paths[0]

		if os.path.isfile(lucs_struct_path):
			pass
		else:
			with gzip.open(os.path.join(lucs_struct_dir,"design.pdb.gz"), 'rb') as f_in:
				with open(os.path.join(lucs_struct_dir,f"{af_design_id}.pdb"), 'wb') as f_out:
					shutil.copyfileobj(f_in, f_out)

		fd_path = os.path.join(lucs_struct_path)

		if args.af_data_dirs != None or args.af_data_parents != None:
			# Load the PyRosetta pose objects for the 5 AF structures
			# try:
			if args.is_colabfold_format:
				af_struct_1, af_struct_2, af_struct_3, af_struct_4, af_struct_5, af_paths = \
					utils.load_colabfold_structs(args, af_dir, af_design_id)
				results_plddts = utils.load_colabfold_plddts(af_dir,af_design_id)
				af_model_names = ['af_model1','af_model2','af_model3','af_model4','af_model5']
			else:
				if args.is_af3:
					af_struct_1, af_struct_2, af_struct_3, af_struct_4, af_struct_5, af_paths = \
						utils.load_af3_structs(af_dir, design_id=af_design_id)
				elif args.is_af3_linux:
					af_struct_1, af_struct_2, af_struct_3, af_struct_4, af_struct_5, af_paths = \
						utils.load_af3_linux_structs(args, af_dir, design_id=af_design_id)
				else:
					if args.use_relaxed_af:
						af_struct_header='relaxed'
						suffix = '_ptm.pdb'
					else:
						af_struct_header='unrelaxed'
						suffix = '_ptm.pdb'
					af_struct_1, af_struct_2, af_struct_3, af_struct_4, af_struct_5, af_paths = \
						utils.load_af_structs(af_dir, header=af_struct_header, suffix=suffix)

				results_plddts = utils.load_af_plddts(af_dir, args, design_id=af_struct_header)
				af_model_names = ['af_model1','af_model2','af_model3','af_model4','af_model5']

				if args.verbose:
					print('Loaded results_plddts.')

		try:
			fastdesign_struct = pose_from_file(fd_path)
			if args.verbose:
				print('FastDesign Sequence and Sequence Length:')
				print(fastdesign_struct.sequence())
				print(len(fastdesign_struct.sequence()))
		except:
			print('\n\nSkipping Design %s -- Failed to Load FastDesign Structure for: %s\n\n' % \
				(lucs_design_id, fd_path))
			continue


		# Load the sequence of the current design.
		if args.use_af_sequence and (args.af_data_dirs != None or args.af_data_parents != None):
			# Use the current designs' AF predictions to gather the design sequence.
			if args.is_colabfold_format:
				af_paths = utils.get_colabfold_struct_paths(args, af_dir, af_design_id)
				af_path = af_paths[0]
			else:
				## I believe I don't need this, as af_paths is loaded.
				# if args.is_af3_linux:
				# 	results_fnames = af_paths
				# if args.af_is_cif:
				# 	results_fnames = [f'_model_{i}.cif' for i in range(1,6)]
				# else:
				# 	results_fnames = [f'_model_{i}_ptm.pdb' for i in range(1,6)]
				
				results_fnames = af_paths

				# Add 'relaxed' or 'unrelaxed' to the starts of the filenames.
				if args.is_af3:
					results_files = [f'{af_design_id}{fname}' for fname in results_fnames]
				elif args.use_relaxed_af:
					results_files = ['relaxed'+fname for fname in results_fnames]
				else:
					results_files = ['unrelaxed'+fname for fname in results_fnames]

				af_path = os.path.join(af_dir, results_files[0])

			sequence, chain_id = utils.pdb_to_seq(af_path)

		else:
			# Use the FastDesign structure to gather the design sequence.
			sequence, chain_id = utils.pdb_to_seq(fd_path)

		sub_dict['sequence'] = sequence

		# Only process sequences containing the specified subsequence
		if args.filter_by_subsequence != '':
			if sequence.endswith(args.filter_by_subsequence) == False:
				print(f'Skipping design {lucs_design_id} because it does not end with the subsequence {args.filter_by_subsequence}.')
				continue

		fd_reslist = utils.get_residue_list_from_pdb(fd_path)
		fd_res_start_idx = fd_reslist[0]
		if args.verbose:
			print(f"Design model residue start index: {fd_res_start_idx}")

		if os.path.exists(os.path.join(lucs_dir,"design_info.json")):
			design_info_path = os.path.join(lucs_dir,"design_info.json")
		elif os.path.exists(os.path.join(lucs_dir,f"{af_design_id}_design_info.json")):
			design_info_path = os.path.join(lucs_dir,f"{af_design_id}_design_info.json")
		else:
			print(f"Design info file not found in {lucs_dir}. Skipping.")

		# Select the non-backbone_remodeled residues in the FastDesign
		# Structure so they can be aligned to the non-backbone_remodeled
		# residues of the Starting Structure
		# This returns PDB-numbered residues
		pdb_fd_bb_remodeled_residues, pdb_fd_non_bb_remodeled_residues = \
			utils.get_remodeled_residues_from_design_info(
				pdb_file = lucs_struct_path,
				design_info_file = design_info_path)

		# Save the backbone-remodeled residues in the sub_dict
		sub_dict['bb_remodeled_residues'] = ' '.join([str(i) for i in pdb_fd_bb_remodeled_residues])

		# Find the helix residues of the given design
		# This list has len(num_lhls)
		pdb_fd_lhl_residues, pdb_fd_helix_residues = utils.get_lhl_residues(
				args,
				design_info_file = design_info_path,
				seq_len = len(fastdesign_struct.sequence()),
				pose=fastdesign_struct,
				use_dssp=args.use_dssp,
				assume_helix=args.assume_helix,
				bb_rem_all_helical=args.bb_rem_all_helical)

		num_lhls = len(pdb_fd_lhl_residues)

		if args.verbose:
			print('Number of LHLs:')
			print(num_lhls)

		# Load lists of LHL 1 and LHL2 residues
		# fd_lhl1_residues, fd_lhl2_residues = utils.load_lhl_res(fd_bb_remodeled_residues, num_lhls=len(helix_residues))

		if args.verbose:
			print('FastDesign LHL residues:')
			for res_list in pdb_fd_lhl_residues:
				print(res_list)
			print('FastDesign Helix residues:')
			for res_list in pdb_fd_helix_residues:
				print(res_list)

		if len(pdb_fd_helix_residues) == 0:
			print(f'No Helical Residues Found for {lucs_design_id}. Skipping RMSD calculations.')
			continue

		if args.af_data_dirs != None or args.af_data_parents != None:
			# Get bb_rem and non_bb_rem indices and pLDDT values. num_models x num_lhls x num_res
			all_lhl_plddts, all_non_bb_rem_plddts = \
				utils.get_lhl_plddt_values(results_plddts, pdb_fd_lhl_residues)
			# all_lhl1_plddts, all_lhl2_plddts, all_non_bb_rem_plddts = \
			# 	utils.get_lhl_plddt_values(results_plddts, fd_lhl1_residues, fd_lhl2_residues, num_lhls=num_lhls)

			# Splitting this data into 5 columns, one for each AF model.
			if args.verbose:
				print('\n\nSub dict before and after loading plddts')
				print(sub_dict)
			
			sub_dict = utils.load_plddts_into_subdict(sub_dict, results_plddts, all_lhl_plddts, all_non_bb_rem_plddts)
		
		if args.verbose:
			print(sub_dict)

		fd_bb_remodeled_residues = utils.convert_pdb_to_pose_resnums(fastdesign_struct, pdb_fd_bb_remodeled_residues, verbose=args.verbose)
		fd_non_bb_remodeled_residues = utils.convert_pdb_to_pose_resnums(fastdesign_struct, pdb_fd_non_bb_remodeled_residues, verbose=args.verbose)
		fd_lhl_residues = []
		for i in range(len(pdb_fd_lhl_residues)):
			fd_lhl_residues.append([])
			fd_lhl_residues[i] = utils.convert_pdb_to_pose_resnums(fastdesign_struct, pdb_fd_lhl_residues[i], verbose=args.verbose)
		fd_helix_residues = []
		for i in range(len(pdb_fd_helix_residues)):
			fd_helix_residues.append([])
			fd_helix_residues[i] = utils.convert_pdb_to_pose_resnums(fastdesign_struct, pdb_fd_helix_residues[i], verbose=args.verbose)


		# Load strings containing residue ranges for PyRosetta
		fd_non_bb_remodeled_residues_string_list, fd_bb_remodeled_residues_string_list = \
			utils.load_pyrosetta_strings(lucs_struct_path, fd_lhl_residues)

		if args.verbose:
			print('FastDesign bb_rem and non-bb_rem residues:')
			print(fd_bb_remodeled_residues_string_list)
			print(fd_non_bb_remodeled_residues_string_list)

		## Calculate the TM score between FastDesign and Starting Structure
		## Using pypi.org/project/tmtools/ to recalculate TM scores.
		fd_ss_tm_score = utils.calc_TM_score(fd_path, args.starting_structure_path)

		# Load the FastDesign to Starting Structure TM score into the sub_dict
		sub_dict['TM_score_fd_to_ss'] = fd_ss_tm_score

		# Calculate Helix RMSDs between FastDesign and Starting Struct

		helix_idxs = [i for i in range(len(fd_helix_residues))]

		if args.verbose:
			print('Helix Indices:')
			print(helix_idxs)

		# If aligning on specified residues
		if args.use_align_residues_file:
			# Returns a list of two lists, containing 1) the starting structure's and 2) the design's residues on which to align.
			# Each sublist contains a list of contiguous residues on which to align, for utils.align_by_residues
			pdb_align_residues = utils.get_design_to_ss_aligned_residues(args, fd_path)

			if args.verbose:
				print('Residues on which to align:')
				print(f'Starting Structure: {pdb_align_residues[0]}')
				print(f'Design Structure: {pdb_align_residues[1]}\n')

			# Convert the starting structure and design models' align residues to PyRosetta pose numbering
			align_residues = [[],[]]
			for i in range(len(pdb_align_residues[0])):
				align_residues[0].append([])
				align_residues[0][i] = utils.convert_pdb_to_pose_resnums(start_struct, pdb_align_residues[0][i], verbose=args.verbose)
			for i in range(len(pdb_align_residues[1])):
				align_residues[1].append([])
				align_residues[1][i] = utils.convert_pdb_to_pose_resnums(fastdesign_struct, pdb_align_residues[1][i], verbose=args.verbose)

			pdb_ss_all_sheet_residues = pdb_align_residues[0] # PDB-numbered sheet residues
			pdb_fd_all_sheet_residues = pdb_align_residues[1]
			ss_all_sheet_residues = align_residues[0] # 0-indexed sheet residues
			fd_all_sheet_residues = align_residues[1]

		# Otherwise, align on non-reshaped residues by default
		else:
			align_residues = None # these are later set to non-bb_rem residues
			pdb_ss_all_sheet_residues = None
			pdb_fd_all_sheet_residues = None
			ss_all_sheet_residues = None
			fd_all_sheet_residues = None

		# Return a list of lists of sheet residues
		# Use PDB-numbered residues, as helix displacement and angle calculations use Biopython to get residue
		# coordinates, and Biopython uses PDB numbering.
		pdb_ss_sheet_residues = utils.get_sheet_residues(start_struct, 
								pdb_start_idx=args.start_struct_res_start_idx,
								strand_nums=None, # Use second beta strand with strand_nums=[1]
								all_sheet_residues=pdb_ss_all_sheet_residues) # ss_align_residues, not None for natural Rossmanns.

		pdb_fd_sheet_residues = utils.get_sheet_residues(fastdesign_struct,
								pdb_start_idx=fd_res_start_idx,
								strand_nums=None, # Use second beta strand with strand_nums=[1]
								all_sheet_residues=pdb_fd_all_sheet_residues) # fd_align_residues, not None for natural Rossmanns
		
		# Align the design and starting structures on their align_residues, if defined
		if align_residues != None:
			# superimpose_poses_by_residues expects a single list of residues on which to align.
			fastdesign_struct, start_struct = utils.superimpose_poses_by_residues(
				fastdesign_struct, 
				[i for lst in align_residues[1] for i in lst], 
				start_struct, 
				[i for lst in align_residues[0] for i in lst], 
				atom_types=['N', 'CA', 'C'])

		# If align_residues is not defined, align structures on their non-remodeled residues.
		else:
			fastdesign_struct, start_struct = utils.superimpose_poses_by_residues(
				fastdesign_struct, 
				fd_non_bb_remodeled_residues, 
				start_struct, 
				fd_non_bb_remodeled_residues, 
				atom_types=['N', 'CA', 'C'])

		all_rmsd_types = [
			pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_heavy_including_O,
			pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_ca]
		all_atom_types_names = ['all_heavy_bb', 'CA']
		helix_rmsd_alignment_atom_types = [['N','CA','C'],['CA']]

		for rmsd_type, atom_types_name, helix_rmsd_atom_types in zip(all_rmsd_types, all_atom_types_names, helix_rmsd_alignment_atom_types):

			fd_ss_helix_rmsd, trimmed_ss_helix_residues, trimmed_fd_helix_residues = utils.calculate_bb_remodeled_region_rmsd(args,
					args.starting_structure_path,
					fd_path,
					helix_idxs=helix_idxs,
					is_af_design=False,
					atom_types=helix_rmsd_atom_types,
					ref_lhl_residues=ss_lhl_residues,
					pose_lhl_residues=fd_lhl_residues,
					ref_helix_residues=ss_helix_residues,
					pose_helix_residues=fd_helix_residues,
					align_residues=align_residues)

			sub_dict['helix_rmsd_fd_to_ss_'+atom_types_name] = fd_ss_helix_rmsd

			if args.verbose:
				print(f'{af_design_id} Helix RMSDs: {atom_types_name}')
				print(fd_ss_helix_rmsd)

			# Calculate Helix RMSD for each individual reshaped helix
			for helix_idx in helix_idxs:
				fd_ss_single_helix_rmsd, trimmed_ss_single_helix_residues, trimmed_fd_single_helix_residues = utils.calculate_bb_remodeled_region_rmsd(args,
						args.starting_structure_path,
						fd_path,
						helix_idxs=[helix_idx],
						is_af_design=False,
						atom_types=helix_rmsd_atom_types,
						ref_lhl_residues=ss_lhl_residues,
						pose_lhl_residues=fd_lhl_residues,
						ref_helix_residues=ss_helix_residues,
						pose_helix_residues=fd_helix_residues,
						align_residues=align_residues)

				sub_dict[f'helix{helix_idx}_rmsd_fd_to_ss_'+atom_types_name] = fd_ss_single_helix_rmsd

				if args.verbose:
					print(f'{af_design_id} Helix {helix_idx} RMSD: '+atom_types_name)
					print(fd_ss_single_helix_rmsd)
					print(f"\n{af_design_id} Trimmed SS and FD Helix Residues:")
					print(' '.join([str(r) for r in trimmed_ss_helix_residues]))
					print(' '.join([str(r) for r in trimmed_fd_helix_residues]))

				# Save lists of trimmed helix residues
				sub_dict[f'trimmed_ss_helix{helix_idx}_residues'] = ' '.join([str(r) for r in trimmed_ss_helix_residues])
				sub_dict[f'trimmed_fd_helix{helix_idx}_residues'] = ' '.join([str(r) for r in trimmed_fd_helix_residues])

			# Save lists of FastDesign helix residues
			sub_dict['fd_helix_residues'] = ' '.join([str(r) for res_list in fd_helix_residues for r in res_list])
			for helix_idx in helix_idxs:
				sub_dict[f'fd_helix{helix_idx}_residues'] = ' '.join([str(r) for r in fd_helix_residues[helix_idx]])

			# Save lists of Starting Structure helix residues
			sub_dict['ss_helix_residues'] = ' '.join([str(r) for res_list in ss_helix_residues for r in res_list])
			for helix_idx in helix_idxs:
				sub_dict[f'ss_helix{helix_idx}_residues'] = ' '.join([str(r) for r in ss_helix_residues[helix_idx]])


			#####
			###  Steph's Helix RMSD functions
			#####
			# 1. Align design and ss poses with utils.steph_align_by_residues
				# - Use lists of strings to select non-bb-rem or align residues 
			# 2. Calculate Helix RMSDs on the aligned poses and their fd_helix_residues and ss_helix_residues with steph_calculate_RMSDs_on_beta_only_helix()
			# 3. Find the longest common helix indices with utils.steph_get_shared_helical_resn()
			# 4. Calculate the Helix RMSDs on the longest common helices with utils.steph_calculate_RMSDs_on_beta_only_helix()
			# Repeat this for AF Helix RMSDs (to design and ss) below.

			# If align_residues is not defined, align structures on their non-remodeled residues.
			if align_residues == None:
				fd_non_bb_rem_align_residues = utils.split_contiguous_residues(fd_non_bb_remodeled_residues)
				ss_non_bb_rem_align_residues = utils.split_contiguous_residues(ss_non_bb_remodeled_residues)

			fd_align_res_str_list = [f'{res[0]}-{res[-1]}' for res in fd_non_bb_rem_align_residues]
			ss_align_res_str_list = [f'{res[0]}-{res[-1]}' for res in ss_non_bb_rem_align_residues]

			if args.verbose:
				print(sub_dict['lucs_location'])
				print('fd_align_res_str_list:')
				print(fd_align_res_str_list)
				print('ss_align_res_str_list:')
				print(ss_align_res_str_list)

			# Align the design and starting structures on their non-remodeled residues
			fastdesign_struct, start_struct = utils.steph_align_by_residues(
				fastdesign_struct, start_struct, 
				fd_align_res_str_list, ss_align_res_str_list)

			# Define lists of strings of the helix residues for the design and starting structures
			# Ensure that each list of helix residues is of the same length, for utils.steph_calculate_RMSDs_on_beta_only_helix()			
			fd_helix_residues_str_list = [f'{res[0]}-{res[-1]}' for res in trimmed_fd_helix_residues]
			ss_helix_residues_str_list = [f'{res[0]}-{res[-1]}' for res in trimmed_ss_helix_residues]
			
			# Calculate Helix RMSDs between FastDesign and Starting Struct
			# This function uses rmsd_protein_bb_heavy_including_O
			# This function also requires that the residue selectors have the same number of residues
			fd_ss_helix_rmsds = utils.steph_calculate_RMSDs_on_beta_only_helix(
				fastdesign_struct, start_struct, 
				fd_helix_residues_str_list, ss_helix_residues_str_list)

			# Save the Helix RMSDs to the sub_dict
			for i, rmsd in enumerate(fd_ss_helix_rmsds):
				sub_dict[f'helix{i}_rmsd_fd_to_ss_bb_heavy_including_O'] = rmsd
				if args.verbose:
					print(f'{af_design_id} Helix {i} RMSD FD to SS: {rmsd}')

			# Find the longest common helix indices between the design and starting structures
			fd_shared_helix_resn, ss_shared_helix_resn = utils.steph_get_shared_helical_resn(
				fastdesign_struct, start_struct, 
				fd_lhl_residues, ss_lhl_residues)

			# Define the shared helix residues as lists of strings
			fd_shared_helix_resn_str_list = [f'{res[0]}-{res[-1]}' if len(res) > 1 else None for res in fd_shared_helix_resn]
			ss_shared_helix_resn_str_list = [f'{res[0]}-{res[-1]}' if len(res) > 1 else None for res in ss_shared_helix_resn]

			# Calculate the Helix RMSDs on the longest common helices
			fd_ss_shared_helix_rmsds = utils.steph_calculate_RMSDs_on_beta_only_helix(
				fastdesign_struct, start_struct, 
				fd_shared_helix_resn_str_list, ss_shared_helix_resn_str_list)

			# Save the shared Helix RMSDs to the sub_dict
			for i, rmsd in enumerate(fd_ss_shared_helix_rmsds):
				sub_dict[f'common_helix{i}_rmsd_fd_to_ss_bb_heavy_including_O'] = rmsd
				if args.verbose:
					print(f'{af_design_id} Common Helix {i} RMSD FD to SS: {rmsd}')

			###
			### End Calculating RMSDs with Steph's Helix RMSD functions
			###

			# Calculate AF RMSDs to Design and Starting Structures
			if args.af_data_dirs != None or args.af_data_parents != None:
				# Select the non_bb_rem residues from the AF
				# structures, then calculate RMSDs between the reshaped residues and
				# store them in dictionaries (rmsd_dict and rmsd_reshaped_residues_dict)

				af_structs = [af_struct_1, af_struct_2, af_struct_3, af_struct_4, af_struct_5]

				for af_idx in range(len(af_structs)):
					af_struct = af_structs[af_idx]
					af_model_name = af_model_names[af_idx]
					
					if args.verbose:
						print("\nCurrent AF structure:")
						print(af_model_name)

					rmsd_atom_types = helix_rmsd_atom_types
					rmsd_atom_types_name = atom_types_name

					af_path = af_paths[af_idx]

					# Calculate TM score between the current AF structure and
					# the Starting Structure
					af_ss_tm_score = \
						utils.calc_TM_score(af_path, args.starting_structure_path)
					# Load the FastDesign to Starting Structure TM score into the sub_dict
					sub_dict['TM_score_ss_to_'+af_model_name] = af_ss_tm_score

					# Calculate TM score between the current AF structure and
					# the FastDesign Structure
					af_fd_tm_score = \
						utils.calc_TM_score(af_path, fd_path)
					# Load the FastDesign to Starting Structure TM score into the sub_dict
					sub_dict['TM_score_fd_to_'+af_model_name] = af_fd_tm_score

					# Align the AF and starting structures on their align_residues, if defined
					if align_residues != None:
						# superimpose_poses_by_residues expects a single list of residues on which to align.
						af_struct, start_struct = utils.superimpose_poses_by_residues(
							af_struct, 
							[i for lst in align_residues[1] for i in lst], 
							start_struct, 
							[i for lst in align_residues[0] for i in lst], 
							atom_types=['N', 'CA', 'C'])

					# If align_residues is not defined, align structures on their non-remodeled residues.
					else:
						af_struct, start_struct = utils.superimpose_poses_by_residues(
							af_struct, 
							fd_non_bb_remodeled_residues, 
							start_struct, 
							fd_non_bb_remodeled_residues, 
							atom_types=['N', 'CA', 'C'])


					# Calculate the Helix Displacements and Angles for AF predictions, relative to the beta sheet residues.
					af_helix_displacements = utils.get_helix_displacements(
							af_struct,
							af_path, # for PDBParser() and DSSP()
							fd_helix_residues,
							chain_id=chain_id,
							struct_res_start_idx=1,
							verbose=args.verbose,
							sheet_residues=pdb_fd_sheet_residues) # sheet_residues defined when calculating helix disp and angle for design to SS above.

					if args.verbose:
						print(f"\n\n\n{af_model_name} Helix Displacements: {af_helix_displacements}\n\n\n")

					# Calculate the helix displacements to 2LV8
					af_helix_displacements_to_ss = utils.get_helix_displacements_to_ss(
							args.starting_structure_path,
							af_path, # for PDBParser() and DSSP()
							pdb_ss_helix_residues,
							pdb_fd_helix_residues,
							ss_chain_id='A',
							design_chain_id='A',
							ss_res_start_idx=args.start_struct_res_start_idx,
							design_res_start_idx=1,
							verbose=args.verbose)

					if args.verbose:
						print(f"\n\n\n{af_model_name} Helix Displacements to Starting Structure: {af_helix_displacements_to_ss}\n\n\n")

					# Instead of calculating sheet_dir_vect with the starting structure, calculate
					# it on the beta sheet of each design model
					sheet_dir_vect = utils.get_strand_dir_vec(af_struct,
							pdb_fd_sheet_residues,
							chain_id='A') # Starting structure chain

					af_helix_angles = utils.get_helix_angles(
							af_struct,
							fd_helix_residues,
							chain_id=chain_id,
							struct_res_start_idx=1,
							verbose=args.verbose,
							sheet_residues=pdb_fd_sheet_residues,
							sheet_dir_vect=sheet_dir_vect)

					if args.verbose:
						print(f"\n\n\n{af_model_name} Helix Angles: {af_helix_angles}\n\n\n")


					# Calculate Helix Angles between AF predictions and the starting structure
					af_helix_angles_to_ss = utils.get_helix_angles_to_ss(
							start_struct,
							af_struct,
							pdb_ss_helix_residues,
							pdb_fd_helix_residues,
							ss_chain_id='A',
							design_chain_id='A',
							ss_struct_res_start_idx=args.start_struct_res_start_idx,
							design_struct_res_start_idx=1,
							verbose=args.verbose)
					
					if args.verbose:
						print(f"\n\n\n{af_model_name} Model Helix Angles to Starting Structure: {af_helix_angles_to_ss}\n\n\n")

					for lhl_idx in range(num_lhls):
						sub_dict[f'helix{lhl_idx}_displacement_{af_model_name}'] = af_helix_displacements[lhl_idx]
						sub_dict[f'helix{lhl_idx}_angle_{af_model_name}'] = af_helix_angles[lhl_idx]
						sub_dict[f'helix{lhl_idx}_displacement_{af_model_name}_to_ss'] = af_helix_displacements_to_ss[lhl_idx]
						sub_dict[f'helix{lhl_idx}_angle_{af_model_name}_to_ss'] = af_helix_angles_to_ss[lhl_idx]

					# Align on non-backbone remodeled residues (FD-AF).		
					fastdesign_struct, af_struct = utils.align_by_residues(args, fastdesign_struct, af_struct,
						fd_non_bb_remodeled_residues_string_list, fd_non_bb_remodeled_residues_string_list)

					# Calculate RMSDs over specific residue sets (FD-AF).
					af_fd_reshaped_rmsd = utils.calculate_RMSD(fastdesign_struct, af_struct, fd_bb_remodeled_residues_string_list,
						rmsd_type=rmsd_type)

					# Save the all-reshaped-residue RMSD between AF and FastDesign
					sub_dict[f'rmsd_reshaped_{atom_types_name}_fd_to_{af_model_name}'] = af_fd_reshaped_rmsd

					for helix_idx in helix_idxs:
						af_fd_lhl_rmsd = utils.calculate_RMSD(fastdesign_struct, af_struct,
							[f"{fd_lhl_residues[helix_idx][0]}-{fd_lhl_residues[helix_idx][-1]}"],
							rmsd_type=rmsd_type)

						sub_dict[f'lhl{helix_idx}_rmsd_{atom_types_name}_fd_to_{af_model_name}'] = af_fd_lhl_rmsd

					# Align on all residues (FD-AF).
					fastdesign_struct, af_struct = utils.align_by_residues(args, fastdesign_struct, af_struct,
						[f"1-{len(fastdesign_struct.sequence())}"], [f"1-{len(af_struct.sequence())}"])

					# Calculate RMSD over all residues (FD-AF).
					all_res_rmsd = utils.calculate_RMSD(fastdesign_struct, af_struct, [f"1-{len(fastdesign_struct.sequence())}"],
						rmsd_type=rmsd_type)

					sub_dict[f'rmsd_all_res_{atom_types_name}_fd_to_{af_model_name}'] = all_res_rmsd

					# Calculate Helix RMSDs between the current AF prediction and FastDesign Struct
					af_fd_helix_rmsd, trimmed_fd_helix_residues, trimmed_af_helix_residues = utils.calculate_bb_remodeled_region_rmsd(args,
							fd_path,
							af_path,
							helix_idxs=helix_idxs,
							is_af_design=True,
							atom_types=helix_rmsd_atom_types,
							ref_lhl_residues=fd_lhl_residues,
							pose_lhl_residues=fd_lhl_residues,
							ref_helix_residues=fd_helix_residues,
							pose_helix_residues=fd_helix_residues)

					sub_dict[f'helix_rmsd_{af_model_name}_to_fd_{atom_types_name}'] = af_fd_helix_rmsd

					if args.verbose:
						print(f"{af_model_name} path: {af_path}")
						print(f'{af_design_id} Helix RMSD AF to FD {atom_types_name}: {af_fd_helix_rmsd}')

					# Calculate Helix RMSD between AF and FD for each individual reshaped helix
					for helix_idx in helix_idxs:
						af_fd_single_helix_rmsd, trimmed_fd_single_helix_residues, trimmed_af_single_helix_residues = utils.calculate_bb_remodeled_region_rmsd(args,
							fd_path,
							af_path,
							helix_idxs=[helix_idx],
							is_af_design=True,
							atom_types=helix_rmsd_atom_types,
							ref_lhl_residues=fd_lhl_residues,
							pose_lhl_residues=fd_lhl_residues,
							ref_helix_residues=fd_helix_residues,
							pose_helix_residues=fd_helix_residues)

						sub_dict[f'helix{helix_idx}_rmsd_{af_model_name}_to_fd_{atom_types_name}'] = af_fd_single_helix_rmsd

						if args.verbose:
							print(f'{af_design_id} Helix {helix_idx} RMSD {af_model_name} to FD {atom_types_name}: {af_fd_single_helix_rmsd}')

					# Calculate Helix RMSDs between the current AF prediction and Starting Struct
					af_ss_helix_rmsd, trimmed_ss_helix_residues, trimmed_af_helix_residues = utils.calculate_bb_remodeled_region_rmsd(args,
							args.starting_structure_path,
							af_path,
							helix_idxs=helix_idxs,
							is_af_design=True,
							atom_types=helix_rmsd_atom_types,
							ref_lhl_residues=ss_lhl_residues,
							pose_lhl_residues=fd_lhl_residues,
							ref_helix_residues=ss_helix_residues,
							pose_helix_residues=fd_helix_residues)

					sub_dict[f'helix_rmsd_{af_model_name}_to_ss_{atom_types_name}'] = af_ss_helix_rmsd

					if args.verbose:
						print(f'{af_design_id} Helix RMSD {af_model_name} to SS {atom_types_name}: {af_ss_helix_rmsd}')

					# Calculate Helix RMSD between AF and SS for each individual reshaped helix
					for helix_idx in helix_idxs:
						af_ss_single_helix_rmsd, trimmed_ss_single_helix_residues, trimmed_af_single_helix_residues = utils.calculate_bb_remodeled_region_rmsd(args,
							args.starting_structure_path,
							af_path,
							helix_idxs=[helix_idx],
							is_af_design=True,
							atom_types=helix_rmsd_atom_types,
							ref_lhl_residues=ss_lhl_residues,
							pose_lhl_residues=fd_lhl_residues,
							ref_helix_residues=ss_helix_residues,
							pose_helix_residues=fd_helix_residues)

						sub_dict[f'helix{helix_idx}_rmsd_{af_model_name}_to_ss_{atom_types_name}'] = af_ss_single_helix_rmsd

						if args.verbose:
							print(f'{af_design_id} Helix {helix_idx} RMSD {af_model_name} to SS {atom_types_name}: {af_ss_single_helix_rmsd}')

					###
					# Use Steph's Alignment and RMSD functions to calculate Helix RMSDs
					# between the current AF prediction and Starting Struct
					###

					# If align_residues is not defined, align structures on their non-remodeled residues.
					if align_residues == None:
						af_non_bb_rem_align_residues = utils.split_contiguous_residues(fd_non_bb_remodeled_residues)
						ss_non_bb_rem_align_residues = utils.split_contiguous_residues(ss_non_bb_remodeled_residues)

					# If the lengths of the AF prediction and FastDesign structures are different, then trim
					# the align residues to the length of the shorter structure.
					if len(fastdesign_struct.sequence()) != len(af_struct.sequence()):
						# Get the length of the shorter structure
						min_len = min(len(fastdesign_struct.sequence()), len(af_struct.sequence()))
						af_non_bb_rem_align_residues[-1] = [res for res in af_non_bb_rem_align_residues[-1] if res <= min_len]

					af_align_res_str_list = [f'{res[0]}-{res[-1]}' for res in af_non_bb_rem_align_residues]
					ss_align_res_str_list = [f'{res[0]}-{res[-1]}' for res in ss_non_bb_rem_align_residues]

					# Align the design and starting structures on their non-remodeled residues
					af_struct, start_struct = utils.steph_align_by_residues(
						af_struct, start_struct, 
						af_align_res_str_list, ss_align_res_str_list)

					# Calculate Helix RMSDs between FastDesign and Starting Struct
					# This function uses rmsd_protein_bb_heavy_including_O
					af_ss_helix_rmsds = utils.steph_calculate_RMSDs_on_beta_only_helix(
						af_struct, start_struct, 
						fd_helix_residues_str_list, ss_helix_residues_str_list)

					# Save the Helix RMSDs to the sub_dict
					for i, rmsd in enumerate(af_ss_helix_rmsds):
						sub_dict[f'helix{i}_rmsd2_{af_model_name}_to_ss_bb_heavy_including_O'] = rmsd
						if args.verbose:
							print(f'{af_design_id} Helix {i} RMSD2 {af_model_name} to SS: {rmsd}')

					# Find the longest common helix indices between the design and starting structures
					# steph_get_shared_helical_resn() expects lists of lists of ints (1-indexed residue numbers)
					af_shared_helix_resn, ss_shared_helix_resn = utils.steph_get_shared_helical_resn(
						af_struct, start_struct, 
						fd_lhl_residues, ss_lhl_residues)

					# Define the shared helix residues as lists of strings
					af_shared_helix_resn_str_list = [f'{res[0]}-{res[-1]}' if len(res) > 1 else None for res in af_shared_helix_resn]
					ss_shared_helix_resn_str_list = [f'{res[0]}-{res[-1]}' if len(res) > 1 else None for res in ss_shared_helix_resn]

					# Calculate the Helix RMSDs on the longest common helices
					af_ss_shared_helix_rmsds = utils.steph_calculate_RMSDs_on_beta_only_helix(
						af_struct, start_struct, 
						af_shared_helix_resn_str_list, ss_shared_helix_resn_str_list)

					# Save the shared Helix RMSDs to the sub_dict
					for i, rmsd in enumerate(af_ss_shared_helix_rmsds):
						sub_dict[f'common_helix{i}_rmsd_{af_model_name}_to_ss_bb_heavy_including_O'] = rmsd
						if args.verbose:
							print(f'{af_design_id} Common Helix {i} RMSD {af_model_name} to SS: {rmsd}')

					###
					# Use Steph's Alignment and RMSD functions to calculate Helix RMSDs
					# between the current AF prediction and FastDesign Struct
					###

					# Align the design and starting structures on their non-remodeled residues
					af_struct, fastdesign_struct = utils.steph_align_by_residues(
						af_struct, fastdesign_struct, 
						af_align_res_str_list, fd_align_res_str_list)

					# Calculate Helix RMSDs between FastDesign and Starting Struct
					# This function uses rmsd_protein_bb_heavy_including_O
					af_fd_helix_rmsds = utils.steph_calculate_RMSDs_on_beta_only_helix(
						af_struct, fastdesign_struct, 
						fd_helix_residues_str_list, fd_helix_residues_str_list)

					# Save the Helix RMSDs to the sub_dict
					for i, rmsd in enumerate(af_fd_helix_rmsds):
						sub_dict[f'helix{i}_rmsd2_{af_model_name}_to_fd_bb_heavy_including_O'] = rmsd
						if args.verbose:
							print(f'{af_design_id} Helix {i} RMSD2 {af_model_name} to FD: {rmsd}')

					# Find the longest common helix indices between the design and starting structures
					af_shared_helix_resn, fd_shared_helix_resn = utils.steph_get_shared_helical_resn(
						af_struct, fastdesign_struct, 
						fd_lhl_residues, fd_lhl_residues)

					# Define the shared helix residues as lists of strings
					af_shared_helix_resn_str_list = [f'{res[0]}-{res[-1]}' if len(res) > 1 else None for res in af_shared_helix_resn]
					fd_shared_helix_resn_str_list = [f'{res[0]}-{res[-1]}' if len(res) > 1 else None for res in fd_shared_helix_resn]

					# Calculate the Helix RMSDs on the longest common helices
					af_fd_shared_helix_rmsds = utils.steph_calculate_RMSDs_on_beta_only_helix(
						af_struct, fastdesign_struct, 
						af_shared_helix_resn_str_list, fd_shared_helix_resn_str_list)

					# Save the shared Helix RMSDs to the sub_dict
					for i, rmsd in enumerate(af_fd_shared_helix_rmsds):
						sub_dict[f'common_helix{i}_rmsd2_{af_model_name}_to_fd_bb_heavy_including_O'] = rmsd
						if args.verbose:
							print(f'{af_design_id} Common Helix {i} RMSD2 {af_model_name} to FD: {rmsd}')

		if args.verbose:
			print(sub_dict)
		sub_dicts.append(sub_dict)

		if args.test_stop != None:
			if args.test_stop == task_idx:
				break
	
	# Load the sub_dicts into a pd DataFrame
	df = pd.DataFrame.from_dict(sub_dicts)
	print(df)

	if args.af_data_dirs != None or args.af_data_parents != None:
		# Add columns to the orderable_df denoting the best AF model w.r.t. the
		# specified AF metric.
		df_col_prefix_values = 'best_af_value_'
		df_col_prefix_models = 'best_af_model_'

		# Prefix to ave_plddt_af_model1
		af_metric = 'ave_plddt_af_'
		af_metric_low_better = False
		df = utils.update_df_with_best_af_model(df, df_col_prefix_values, 
			df_col_prefix_models, af_metric, af_metric_low_better, add_best_af_metrics=True)

		# Prefix to rmsd_reshaped_CA_fd_to_af_model1
		af_metric = 'rmsd_reshaped_CA_fd_to_af_'
		af_metric_low_better = True
		df = utils.update_df_with_best_af_model(df, df_col_prefix_values, 
			df_col_prefix_models, af_metric, af_metric_low_better, add_best_af_metrics=True)

		# Prefix to rmsd_reshaped_CA_fd_to_af_model1
		af_metric = 'rmsd_reshaped_all_heavy_bb_fd_to_af_'
		af_metric_low_better = True
		df = utils.update_df_with_best_af_model(df, df_col_prefix_values, 
			df_col_prefix_models, af_metric, af_metric_low_better, add_best_af_metrics=True)

	# Update df with LUCS metrics
	df = utils.update_df_with_LUCS_metrics(df)

	# Edit the outfile to include the args.task_id, to save args.num_tasks
	# different dfs that can be combined later.
	outfile = args.outfile.replace('.csv', f'_{args.task_id}.csv')

	# Save the dataframe in a csv file
	df.to_csv(os.path.join(args.outfolder, outfile))

	print('Saved Dataframe: %s' % os.path.join(args.outfolder, outfile))

