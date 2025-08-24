import os
import pandas as pd
import argparse

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

# Import utility functions for degree_reshaped.py
import utils_degree_reshaped as utils



if __name__ == "__main__":
	# Parse command line arguments.
	parser = argparse.ArgumentParser(
		description='Calculate RMSD between LUCS models and Rosetta Relaxed models.',
		formatter_class=argparse.RawDescriptionHelpFormatter)

	# LUCS inputs
	parser.add_argument('--df_path', type=str, required=True,
			help='Path to a directory containing LUCS designs.')

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

	# Debugging arguments
	parser.add_argument('--verbose', action='store_true',
			help='Print extra print statements.')
	parser.add_argument('--test_stop', type=int, required=False, default=None,
			help='Number of test calculations to perform.')


	args = parser.parse_args()

	# Make output folder
	os.makedirs(args.outfolder, exist_ok=True)

	# Load the input dataframe
	df = pd.read_csv(args.df_path)

	design_idx = 0

	# Load a DataFrame with all the degree of reshaping and confidence data.
	# These data will be held in a list of dictionaries.
	sub_dicts = []

	for idx, row in df.iterrows():
		design_idx += 1
		if design_idx % args.num_tasks != args.task_id - 1:
			continue

		if args.verbose:
			print(f'Processing design {row["design_id"]}')

		lucs_path = os.path.join(row['original_lucs_location'],"unzipped_design.pdb")
		lucs_struct = pose_from_file(lucs_path)

		relax_path = os.path.join(row['lucs_location'],f"{row['design_id']}.pdb")
		relax_struct = pose_from_file(relax_path)

		# sub_dict will become a row in the output df
		sub_dict = {}
		sub_dict['design_id'] = row['design_id']
		sub_dict['af2_design_id'] = row['af2_design_id']
		sub_dict['lucs_design_id'] = row['lucs_design_id']
		sub_dict['af2_location'] = row['af2_location']
		sub_dict['lucs_location'] = row['lucs_location']

		lucs_res_start_idx = 1

		# Select the non-backbone_remodeled residues in the FastDesign
		# Structure so they can be aligned to the non-backbone_remodeled
		# residues of the Starting Structure
		lucs_bb_remodeled_residues, lucs_non_bb_remodeled_residues = \
			utils.get_remodeled_residues_from_design_info(
				design_info_file = os.path.join(row['original_lucs_location'],"design_info.json"), 
				res_start_idx = lucs_res_start_idx,
				seq_len = len(lucs_struct.sequence()))

		# Save the backbone-remodeled residues in the sub_dict
		sub_dict['bb_remodeled_residues'] = ' '.join([str(i) for i in lucs_bb_remodeled_residues])

		# Load lists of LHL 1 and LHL2 residues
		lucs_lhl1_residues, lucs_lhl2_residues = utils.load_lhl_res(lucs_bb_remodeled_residues)

		if args.verbose:
			print('LUCS LHL1 and LHL2 residues:')
			print(lucs_lhl1_residues)
			print(lucs_lhl2_residues)

		# Load strings containing residue ranges for PyRosetta
		lucs_non_bb_remodeled_residues_string_list, lucs_bb_remodeled_residues_string_list = \
			utils.load_pyrosetta_strings(len(lucs_struct.sequence()), [lucs_lhl1_residues, lucs_lhl2_residues])

		# all_rmsd_types = [
		# 	pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_heavy_including_O,
		# 	pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_ca]
		# all_atom_types_names = ['all_heavy_bb', 'CA']
		# helix_rmsd_alignment_atom_types = [['N','CA','C'],['CA']]

		all_rmsd_types = [
			pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_heavy_including_O]
		all_atom_types_names = ['all_heavy_bb']
		helix_rmsd_alignment_atom_types = [['N','CA','C']]

		for rmsd_type, atom_types_name, helix_rmsd_atom_types in zip(all_rmsd_types, all_atom_types_names, helix_rmsd_alignment_atom_types):

			# For Reshaped Region RMSDs, align on all non-reshaped residues.
			utils.align_by_residues(lucs_struct, relax_struct, lucs_non_bb_remodeled_residues_string_list)

			# Calculate RMSDs over specific residue sets (FD-AF2).
			relax_lucs_reshaped_rmsd = utils.calculate_RMSD(lucs_struct, relax_struct, lucs_bb_remodeled_residues_string_list,
				rmsd_type=rmsd_type)

			sub_dict['rmsd_reshaped_lucs_to_relax_%s'%(atom_types_name)] = relax_lucs_reshaped_rmsd

			# Align on all residues.
			utils.align_by_residues(lucs_struct, relax_struct, [f"1-{len(lucs_struct.sequence())}"])

			# Calculate RMSD over all residues (FD-AF2).
			all_res_rmsd = utils.calculate_RMSD(lucs_struct, relax_struct, [f"1-{len(lucs_struct.sequence())}"],
				rmsd_type=rmsd_type)

			sub_dict['rmsd_all_res_lucs_to_relax_%s'%(atom_types_name)] = all_res_rmsd

		if args.verbose:
			print('Current Sub Dict:')
			print(sub_dict)

		sub_dicts.append(sub_dict)

		if args.test_stop != None:
			if args.test_stop < design_idx:
				break

	# Load the sub_dicts into a pd DataFrame
	df = pd.DataFrame.from_dict(sub_dicts)
	print(df)

	# Edit the outfile to include the args.task_id, to save args.num_tasks
	# different dfs that can be combined later.
	outfile = args.outfile.replace('.csv', f'_{args.task_id}.csv')

	# Save the dataframe in a csv file
	df.to_csv(os.path.join(args.outfolder, outfile))

	print('Saved Dataframe: %s' % os.path.join(args.outfolder, outfile))

