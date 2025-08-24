"""
Ben Orr
2.20.24

analyze_finetuned_predictions.py

Calculate RMSDs between classic and fine-tuned AF2 predictions to their ground truth structures.  Load and
save a dataframe with the fine-tuned model's predictions' RMSDs to the ground truth structures, their
confidence metrics, and their paths to the PDB files for their classic AF2, fine-tuned AF2, and ground truth
structures.

Usage:
	calc_checkpointed_models_metrics.py --job_name <job_name> --predictions <AF2_prediction_folder>
		--ground_truth <ground_truth_folder1> [ <ground_truth_folder2> ... ]
		--model_names <classic_model_name> <fine_tuned_model_name> --outfolder_parent <output_folder_parent>
		[ --reshaped_region_rmsd --reshaped_residue_confidence --design_info_folders <design_info_folder1> [ <design_info_folder2> ... ]]

Inputs:

	job_name: Name for the job which will be used in the output .png files.

	predictions: Path to a folder containing classic and fine-tuned AF2 predictions (.pdb files), as well as AF2 confidence scores (.npy files).

	ground_truth: A folder (or folders) containing PDB files of the ground truth structures (e.g., design models) of the corresponding
		AF2 predictions.  The PDB headers in the ground truth files must also appear in the headers of the AF2 output files.

	model_names: The classic and fine-tuned AF2 model names (as they appear in the headers of the prediction output files).
		The classic model name should be provided first, followed by the fine-tuned model name.

	outfolder_parent: Output files are written to outfolder_parent.

	reshaped_helix_rmsd: Calculate Reshaped Helix RMSD to Rosetta values for LUCS designs.

	reshaped_residue_confidence: Calculate the pAE values between the bb_remodeled_residues and non-bb_remodeled_residues and
			pLDDT values of the bb_remodeled_residues (this residue set is found in the bb_remodeled_residues column
			of the stability_data dataframe(s)). Uses the default pLDDT and pAE thresholds or the provided thresholds.

	design_info_folders: If using the --reshaped_region_rmsd flag, then a path to a folder (or folders) containing LUCS designs'
			<design_id>_design_info_folder.json files must be provided with this flag. These <design_id>_design_info_folder.json
			files must contain a linker_ids key, which is used to find the helical residues for each design.


Outputs:

	A dataframe containing AF2 prediction (classic and fine-tuned) to ground truth RMSDs, confidence values for both AF2 models for
	each example, and paths to PDB files in outfolder_parent/job_name.

"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted

from Bio.PDB import *
from Bio.PDB.QCPSuperimposer import QCPSuperimposer

import pyrosetta
from pyrosetta import rosetta
pyrosetta.init()

## Writing my own function to calculate RMSD
def calc_RMSD(ref_coords, coords):
	"""
	Takes num_points x num_dims (num_dims=3 for 3d coords) arrays. Calculates the
	displacement between each pair of points, then divides the sum of these
	displacements by the number of points to get the RMSD
	"""
	total_disp = 0
	for idx in range(len(ref_coords)):
		disp = 0
		for dim in range(len(ref_coords[idx])):
			disp += (ref_coords[idx][dim] - coords[idx][dim])**2
		total_disp += disp
	total_disp = total_disp / len(ref_coords)
	total_disp = total_disp**(1/2)
	return total_disp


def get_RMSD(ref_struct, 
			 moving_struct, 
			 atom_types, 
			 ref_align_res_subset, 
			 ref_rmsd_res_subset,
			 moving_align_res_subset, 
			 moving_rmsd_res_subset,
			 ref_chain='A'):
	"""
	Align two models based on a subset of residues, and calcualte the RMSD on the
	atom_types on the rmsd_res_subset residues.

	Res subsets are PDB numbering, since this is what Biopython uses.
	"""
	ref_model = ref_struct[0]
	ref_chain = ref_model[ref_chain]
	moving_model = moving_struct[0]
	moving_chain = moving_model["A"]
	sup = Superimposer()	

	# Load lists of Atom objects that will be used to align
	ref_align_atoms, moving_align_atoms = [], []
	for ref_res_idx, moving_res_idx in zip(ref_align_res_subset, moving_align_res_subset):
		for atom_type in atom_types: # ex. 'CA'
			moving_added = False
			try:
				moving_align_atoms.append(moving_chain[moving_res_idx][atom_type]) # in case the moving atoms cannot be found (as is the case with 170183.pdb from the Round 2 RO2 order), attempt adding these first
				moving_added = True
				ref_align_atoms.append(ref_chain[ref_res_idx][atom_type])
			except:
				print(f'\n\nCould not load moving_rmsd_coords {moving_res_idx} (or ref_rmsd_coords {ref_res_idx}) {atom_type} for alignment in get_RMSD(). Skipping this atom.')
				if moving_added:
					moving_align_atoms.pop()

	sup.set_atoms(ref_align_atoms, moving_align_atoms)
	# Apply the superimposition to the moving_atoms
	sup.apply(moving_struct.get_atoms())

	# Load lists of coordinates that will be used to calc RMSD
	# This must be run after the transformation is applied to the moving_struct
	ref_rmsd_coords, moving_rmsd_coords = [], []
	for ref_res_idx, moving_res_idx in zip(ref_rmsd_res_subset, moving_rmsd_res_subset):
		for atom_type in atom_types: # ex. 'CA'
			moving_added = False
			try:
				moving_rmsd_coords.append(moving_chain[moving_res_idx][atom_type].get_coord()) # in case the moving atoms cannot be found (as is the case with 170183.pdb from the Round 2 RO2 order), attempt adding these first
				moving_added = True
				ref_rmsd_coords.append(ref_chain[ref_res_idx][atom_type].get_coord())
			except:
				print(f'\n\nCould not load moving_rmsd_coords {moving_res_idx} (or ref_rmsd_coords {ref_res_idx}) {atom_type} for alignment in get_RMSD(). Skipping this atom.')
				if moving_added:
					moving_rmsd_coords.pop()

	return calc_RMSD(ref_rmsd_coords, moving_rmsd_coords)

def sns_plot(x_vals, y_vals, x_name, y_name, job_name, outfolder):
	### Plot with seaborn
	sns.set_theme()
	fig = plt.figure()
	ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
	ax.set_xlabel('%s\nRMSD to Rosetta' % x_name) #, fontsize=15)
	ax.set_ylabel('%s RMSD to Rosetta' % y_name, fontsize=15)
	ax.set_title('RMSD to Rosetta for Classic and\nFine-Tuned AF2 models', fontsize=15)
	values = np.vstack([x_vals, y_vals])
	kernel = stats.gaussian_kde(values)(values)
	sns.scatterplot(x=x_vals, y=y_vals, c=kernel, cmap='viridis')
	lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
	        np.max([ax.get_xlim(), ax.get_ylim()])]
	ax.plot(lims, lims, '--', alpha=0.5, zorder=100)
	ax.set_aspect('equal')
	plt.tight_layout()
	plt.savefig(os.path.join(outfolder,'%s_rmsds.png'%(job_name)), dpi=300)
	print('Saved figure: %s.png' % (os.path.join(outfolder,'%s_rmsds'%(job_name))))

def compare_rmsds(list_of_dicts, job_name, outfolder):
	model1_name = list_of_dicts[0]['classic_name']
	model2_name = list_of_dicts[0]['ft_name']
	num_model1_better,num_model2_better=0,0
	model1_rmsds, model2_rmsds = [], []
	for data_dict in list_of_dicts:
		if data_dict['classic_name'] == model1_name:
			model1_rmsd = data_dict['classic_gt_rmsd']
			model2_rmsd = data_dict['ft_gt_rmsd']
		else:
			model1_rmsd = data_dict['ft_gt_rmsd']
			model2_rmsd = data_dict['classic_gt_rmsd']
		model1_rmsds.append(model1_rmsd)
		model2_rmsds.append(model2_rmsd)
		if model1_rmsd < model2_rmsd:
			num_model1_better+=1
		else:
			num_model2_better+=1

	print('%s has lower RMSD to Rosetta for %d structures' % (model1_name, num_model1_better))
	print('%s has lower RMSD to Rosetta for %d structures' % (model2_name, num_model2_better))

	## Always plot with fine-tuned model on x-axis
	if model1_name=='model_2_ptm':
		sns_plot(model2_rmsds, model1_rmsds, model2_name, model1_name, job_name, outfolder)
	else:
		sns_plot(model1_rmsds, model2_rmsds, model1_name, model2_name, job_name, outfolder)

def compare_df_rmsds(df, cutoff=0):
	model1_name = df.iloc[0]['classic_name']
	model2_name = df.iloc[0]['ft_name']
	num_model1_better,num_model2_better=0,0
	model1_rmsds, model2_rmsds = [], []
	for idx, row in df.iterrows():
		if row['classic_name'] == model1_name:
			model1_rmsd = row['classic_gt_rmsd']
			model2_rmsd = row['ft_gt_rmsd']
		else:
			model1_rmsd = row['ft_gt_rmsd']
			model2_rmsd = row['classic_gt_rmsd']
		model1_rmsds.append(model1_rmsd)
		model2_rmsds.append(model2_rmsd)
		if model1_rmsd+cutoff < model2_rmsd:
			num_model1_better+=1
		elif model1_rmsd > model2_rmsd+cutoff:
			num_model2_better+=1

	print('%s lower RMSD to Rosetta = %d (%.1f) Cutoff=%.1f' % \
		(model1_name, num_model1_better, num_model1_better/(num_model1_better+num_model2_better), cutoff))
	print('%s lower RMSD to Rosetta = %d (%.1f) Cutoff=%.1f' % \
		(model2_name, num_model2_better, num_model2_better/(num_model1_better+num_model2_better), cutoff))

	# Should've used this condition when making the RMSD dfs...
	if model1_name=='model_2_ptm':
		print('model 1 was model_2_ptm')
		return num_model1_better, num_model2_better, model1_rmsds, model2_rmsds
	else:
		return num_model2_better, num_model1_better, model2_rmsds, model1_rmsds

def load_confidence_scores(new_dict,af2_folder,model1_file,model2_file):
	"""
	Load plddt, pae, and ptm scores for the classic and ft AF2 models
	"""
	classic_plddt_path = os.path.join(af2_folder, model1_file.replace('.pdb','_plddt.npy'))
	classic_pae_path = os.path.join(af2_folder, model1_file.replace('.pdb','_predicted_aligned_error.npy'))
	classic_ptm_path = os.path.join(af2_folder, model1_file.replace('.pdb','_ptm.npy'))
	ft_plddt_path = os.path.join(af2_folder, model2_file.replace('.pdb','_plddt.npy'))
	ft_pae_path = os.path.join(af2_folder, model2_file.replace('.pdb','_predicted_aligned_error.npy'))
	ft_ptm_path = os.path.join(af2_folder, model2_file.replace('.pdb','_ptm.npy'))

	paths=[classic_plddt_path,classic_pae_path,classic_ptm_path,ft_plddt_path,ft_pae_path,ft_ptm_path]
	keys=[('classic_plddt','classic_plddt_ave'),('classic_pae','classic_pae_ave'),
		('classic_ptm','classic_ptm_ave'),('ft_plddt','ft_plddt_ave'),
		('ft_pae','ft_pae_ave'),('ft_ptm','ft_ptm_ave')]
	for path, key in zip(paths, keys):
		if not os.path.exists(path):
			new_dict[key[0]] = ''
			new_dict[key[1]] = -999
		else:
			try:
				data = np.load(path, allow_pickle=True)
				if 'ptm' in key[1]:
					str_data = str(data)
				else:
					str_data = ' '.join([str(val).replace('\n','') for val in data])
				new_dict[key[0]] = str_data
				new_dict[key[1]] = np.mean(data)
			except:
				print(f'Data from {path} could not be loaded. Skipping these confidence scores.')
				new_dict[key[0]] = ''
				new_dict[key[1]] = -999

	return new_dict

# Helix RMSD functions (these functions use pyrosetta):
def load_design(design_path, bb_remodeled_residues=None, design_info_file=None):
	'''Load a design.
	Return:
		pose_design, pose_to_compare, pose_to_compare_type, bb_remodeled_residues, bb_fixed_residues
	'''
	pose_design = rosetta.core.import_pose.pose_from_file(os.path.join(design_path))

	if bb_remodeled_residues == None:
		with open(design_info_file, 'r') as f:
			bb_remodeled_residues_all = sorted(json.load(f)['bb_remodeled_residues'])
	else:
		# bb_remodeled_residues_all is a flattened bb_remodeled_residues list.
		# bb_remodeled_residues_all = [i for sublist in bb_remodeled_residues for i in sublist]
		bb_remodeled_residues_all = bb_remodeled_residues

	# Separate remodeled residues into segments 
	separators = [i + 1 for i in range(len(bb_remodeled_residues_all) - 1) if bb_remodeled_residues_all[i] + 1 != bb_remodeled_residues_all[i + 1]] 
	separators = [0] + separators + [len(bb_remodeled_residues_all)]

	bb_remodeled_residues = []
	for i in range(len(separators) - 1):
		bb_remodeled_residues.append(list(range(bb_remodeled_residues_all[separators[i]], 
			bb_remodeled_residues_all[separators[i + 1] - 1] + 1)))

	bb_fixed_residues = [i for i in range(1, pose_design.size() + 1) if not (i in bb_remodeled_residues_all)]

	return pose_design, bb_remodeled_residues, bb_fixed_residues

def xyzV_to_np_array(xyz):
	return np.array([xyz.x, xyz.y, xyz.z])

def np_array_to_xyzV(a):
	return rosetta.numeric.xyzVector_double_t(a[0], a[1], a[2])

def np_array_to_xyzM(a):
	return rosetta.numeric.xyzMatrix_double_t.rows(
			a[0][0], a[0][1], a[0][2],
			a[1][0], a[1][1], a[1][2],
			a[2][0], a[2][1], a[2][2])

def get_backbone_points(pose, residues, atom_types=['N', 'CA', 'C']):
	'''Get backbone points for residues in a pose.'''
	points = []

	for res in residues:
		for atom in atom_types:
			points.append(xyzV_to_np_array(pose.residue(res).xyz(atom)))

	return points

def get_superimpose_transformation(P1, P2):
	'''Get the superimpose transformation that transfoms a list of
	points P1 to another list of points P2.'''
	if len(P1) != len(P2):
		raise Exception("Sets to be superimposed must have same number of points.")

	com1 = np.mean(P1, axis=0)
	com2 = np.mean(P2, axis=0)

	R = np.dot(np.transpose(np.array(P1) - com1), np.array(P2) - com2)
	V, S, W = np.linalg.svd(R)

	if (np.linalg.det(V) * np.linalg.det(W)) < 0.0:
		V[:, -1] = -V[:, -1]

	M = np.transpose(np.array(np.dot(V, W)))

	return M, com2 - np.dot(M, com1)

def superimpose_poses_by_residues(pose_source, residues_source, pose_target, residues_target, atom_types=['N', 'CA', 'C']):
	'''Superimpose residues in a source pose into residues in a target pose.
	Only backbone atoms are used for the superimposition.
	'''
	assert(len(residues_source) == len(residues_target))

	# Get the points to be superimposed

	points_source = get_backbone_points(pose_source, residues_source)
	points_target = get_backbone_points(pose_target, residues_target)

	# Get the rigid body transformation

	M, t = get_superimpose_transformation(points_source, points_target)

	# Transform the source pose

	pose_source.apply_transform_Rx_plus_v(np_array_to_xyzM(M), 
			np_array_to_xyzV(t))

def get_helix_of_lhl_unit(pose, lhl_start, lhl_stop):
	'''Get the helix part of a LHL unit.
	Return a pair of (start, stop).
	'''
	dssp_str = rosetta.core.scoring.dssp.Dssp(pose).get_dssp_secstruct()

	h_start = lhl_start

	while h_start < lhl_stop and dssp_str[h_start - 1] != 'H':
		h_start += 1

	h_stop = h_start

	while h_stop < lhl_stop and dssp_str[h_stop] == 'H':
		h_stop += 1

	return h_start, h_stop

def calc_backbone_RMSD(pose1, residues1, pose2, residues2):
	'''Calculate backbone RMSD between two poses for specific positions.'''
	assert(len(residues1) == len(residues2))

	def RMSD(points1, points2):
		'''Calculate RMSD between two lists of numpy points.'''
		diff = [points1[i] - points2[i] for i in range(len(points1))]

		## Ben 4.24.23: I believe that some structures don't have any
		## helical residues identified.
		print('Number of helix residues identified = %d'%len(diff))

		return np.sqrt(sum(np.dot(d, d) for d in diff) / len(diff))

	points1 = get_backbone_points(pose1, residues1)
	points2 = get_backbone_points(pose2, residues2)

	return RMSD(points1, points2)

# def calc_helix_RMSD(data_dict, args, model_type='classic'):
# 	"""
# 	Calculate and return the average helix RMSD between
# 	LUCS designs and classic/fine-tuned AF2 predictions.
# 	"""
# 	design_info_file = os.path.join(args.design_info_folder, str(data_dict['name'])+'_design_info.json')

# 	# Load the reference structure
# 	ref_design, ref_pose_to_compare, ref_pose_to_compare_type, ref_bb_remodeled_residues, ref_bb_remodeled_residues_all, ref_bb_fixed_residues = \
# 		load_design(data_dict['ground_truth_path'], design_info_file)

# 	# Load the LHLR structure
# 	pose_design, pose_to_compare, pose_to_compare_type, bb_remodeled_residues, bb_remodeled_residues_all, bb_fixed_residues \
# 		= load_design(data_dict['%s_path'%model_type], design_info_file)

# 	# Superimpose by the non-reshaped residues
# 	superimpose_poses_by_residues(pose_design, bb_fixed_residues, ref_pose_to_compare, ref_bb_fixed_residues)

# 	# Find the mapped residues between the two structures

# 	mapped_residues_i = []
# 	mapped_residues_j = []

# 	## Ben 4.21.23: Loop over all redesigned LHLs (2 LHLs for the RO2 design task)
# 	for k in range(len(bb_remodeled_residues)):
# 		seg_i = bb_remodeled_residues[k]
# 		# seg_j = all_bb_remodeled_residues[j][k]
# 		seg_j = ref_bb_remodeled_residues[k]

# 		# Get the helix part of the LHL unit

# 		helix_i = get_helix_of_lhl_unit(pose_design, min(seg_i), max(seg_i))
		    
# 		helix_j = get_helix_of_lhl_unit(ref_pose_to_compare, min(seg_j), max(seg_j))

# 		len_comp = min(helix_i[1] - helix_i[0], helix_j[1] - helix_j[0])

# 		h_mid_start1 = (sum(helix_i) - len_comp) // 2
# 		h_mid_start2 = (sum(helix_j) - len_comp) // 2

# 		mapped_residues_i += [h_mid_start1 + l for l in range(len_comp)]
# 		mapped_residues_j += [h_mid_start2 + l for l in range(len_comp)]

# 	rmsd = calc_backbone_RMSD(ref_pose_to_compare, mapped_residues_j, pose_design, mapped_residues_i)

# 	return rmsd

### End Helix RMSD functions.

### Ben Orr 2.20.24: New Helix RMSD functions (which use the linker_ids in the design_info.json
### files to find helical residues, instead of DSSP)
def get_lhl_residues(
	design_info_file, 
	res_start_idx,
	seq_len):

	# Load the list of bb_remodeled_residues
	with open(design_info_file) as design_info_json:
		design_info = json.load(design_info_json)
		bb_remodeled_residues = design_info['bb_remodeled_residues']

	# Load the list of linker_ids
	with open(design_info_file) as design_info_json:
		design_info = json.load(design_info_json)
		try:
			linker_ids = design_info['linker_ids']
		except:
			# If there are no linker_ids keys in the design_info.json (as in the Round 2 RO2 order), then assume linkers are length 5
			linker_ids = None

	# Separate contingous LHLs
	lhl_residues = []
	prev_stop = 0
	for i in range(len(bb_remodeled_residues)-1):
		if bb_remodeled_residues[i] != bb_remodeled_residues[i+1] - 1:
			lhl_residues.append(bb_remodeled_residues[prev_stop:i+1])
			prev_stop = i + 1
	lhl_residues.append(bb_remodeled_residues[prev_stop:])

	# Load a list of bb_remodeled helix residue indices.
	helix_residues = []
	if linker_ids != None:
		for i in range(len(linker_ids)):
			l_d = linker_ids[i]
			# Use LUCS linker database nomenclature to get linker lengths
			front_loop_len = int(l_d['front_linker_id'].split('_non_redundant.json')[0][-1]) # 1-digit loop lengths
			back_loop_len = int(l_d['back_linker_id'].split('_non_redundant.json')[0][-1])

			# lhl_residues indices match those of linker_ids
			helix_residues.append(lhl_residues[i][front_loop_len:-1*back_loop_len])
		
	else:
		# 2 LHLs in RO2 designs.
		for i in range(2):
			# If there are no linker_ids keys in the design_info.json (as in the Round 2 RO2 order), then assume linkers are length 5
			helix_residues.append(lhl_residues[i][5:-5])

	return lhl_residues, helix_residues

def calculate_bb_remodeled_region_rmsd(reference_path,
		design_path,
		helix_idxs=[0,1],
		is_af2_design=False,
		atom_types=['N', 'CA', 'C'],
		ref_lhl_residues=None,
		pose_lhl_residues=None,
		ref_helix_residues=None,
		pose_helix_residues=None):
	'''Calculate the backbone RMSDs of the remodeled region.
	Return a two dimensional list where list[i][j] element
	is the RMSD between design i and experimentally solved
	or lowest scoring model j.
	Return:
		rmsds, pose_to_compare_types

	Ben 5.10.23:
		Calculate the Helix RMSDs between reference and design structures.
		The reference will be either 2lv8 or a LUCS design, and the design will
		be either a LUCS design or an AF2 model.

		bb_remodeled_residues correspond to the design_path design.

		Helix Residues is a list of lists, the first list having
			the length of the number of LHLs, and the nested list
			having the helical residues for each LHL.
	''' 

	# Load the designs

	if ref_lhl_residues != None:
		ref_bb_remodeled_residues = [i for lhl in ref_lhl_residues for i in lhl]
		ref_design, ref_bb_remodeled_residues, ref_bb_fixed_residues \
			= load_design(reference_path, bb_remodeled_residues=ref_bb_remodeled_residues)
	else:
		ref_design, ref_bb_remodeled_residues, ref_bb_fixed_residues \
			= load_design(reference_path, bb_remodeled_residues=None)

	if is_af2_design:
		# Provide the bb_remodeled_residues for the AF2 models.
		pose_bb_remodeled_residues = [i for lhl in pose_lhl_residues for i in lhl]
		pose_design, all_bb_remodeled_residues, all_bb_fixed_residues \
			= load_design(design_path, bb_remodeled_residues=pose_bb_remodeled_residues)
	elif pose_lhl_residues != None:
		pose_bb_remodeled_residues = [i for lhl in pose_lhl_residues for i in lhl]
		pose_design, all_bb_remodeled_residues, all_bb_fixed_residues \
			= load_design(design_path, bb_remodeled_residues=pose_bb_remodeled_residues)
	else:
		pose_design, all_bb_remodeled_residues, all_bb_fixed_residues \
			= load_design(design_path, bb_remodeled_residues=None)

	# print('\n\nBen Orr 1.10.24: all_bb_remodeled_residues and ref_bb_remodeled_residues:')
	# print(all_bb_remodeled_residues)
	# print(ref_bb_remodeled_residues)

	# Superimpose design i to reference design

	if args.verbose:
		print('all_bb_fixed_residues and ref_bb_fixed_residues:')
		print(all_bb_fixed_residues)
		print(ref_bb_fixed_residues)

	try:
		superimpose_poses_by_residues(pose_design, all_bb_fixed_residues, ref_design, ref_bb_fixed_residues, atom_types=atom_types)
	except:
		print('Pose and Ref bb_fixed_residues have different lengths for alignment in calculate_bb_remodeled_region_rmsd()::superimpose_poses_by_residues().\n\
Using Pose residues for alignment (as Pose and Ref have same sequence)')
		if len(all_bb_fixed_residues) < len(ref_bb_fixed_residues):
			superimpose_poses_by_residues(pose_design, all_bb_fixed_residues, ref_design, all_bb_fixed_residues, atom_types=atom_types)
		else:
			superimpose_poses_by_residues(pose_design, ref_bb_fixed_residues, ref_design, ref_bb_fixed_residues, atom_types=atom_types)

	# Find the mapped residues between the two structures

	mapped_residues_i = []
	mapped_residues_j = []

	new_helix_residues = []

	## Ben 3.1.23: Loop over all redesigned LHLs (2 LHLs for the RO2 designs)
	for k in helix_idxs:
		seg_i = all_bb_remodeled_residues[k]
		# seg_j = all_bb_remodeled_residues[j][k]
		seg_j = ref_bb_remodeled_residues[k]

		# print('\n\nBen Orr 1.10.24: seg_i and seg_j before trimming:')
		# print(seg_i)
		# print(seg_j)

		# Get the helix part of the LHL unit

		if args.verbose:
			print('Pose Helix Residues')
			print(pose_helix_residues)

		# If the pose helix residues are provided (as a list of lists)
		if pose_helix_residues != None:
			helix_i = (pose_helix_residues[k][0], pose_helix_residues[k][-1]) # tuple of start and stop indices.
		else:
			helix_i = get_helix_of_lhl_unit(pose_design, min(seg_i), max(seg_i))
		
		# If the reference helix residues are provided (as a list of lists)
		if ref_helix_residues != None:
			helix_j = (ref_helix_residues[k][0], ref_helix_residues[k][-1]) # tuple of start and stop indices.
		else:
			helix_j = get_helix_of_lhl_unit(ref_design, min(seg_j), max(seg_j))

		len_comp = min(helix_i[1] - helix_i[0], helix_j[1] - helix_j[0])

		h_mid_start1 = (sum(helix_i) - len_comp) // 2
		h_mid_start2 = (sum(helix_j) - len_comp) // 2

		mapped_residues_i += [h_mid_start1 + l for l in range(len_comp)]
		mapped_residues_j += [h_mid_start2 + l for l in range(len_comp)]

		# Store and return the design pose's helix residues
		new_helix_residues.append(helix_i)

	# print('\n\nBen Orr 1.10.24: mapped_residues_j and mapped_residues_i:')
	# print(mapped_residues_j)
	# print(mapped_residues_i)

	# Calculate RMSD over all helix residues
	rmsd = calc_backbone_RMSD(ref_design, mapped_residues_j, pose_design, mapped_residues_i)

	# print(f'Helix RMSD: {rmsd}')

	return rmsd, new_helix_residues

### End New Helix RMSD Functions.

def get_reshaped_confidence(data_dict, args, model_type='classic', bb_remodeled_residues=None, design_info_file=None):
	"""
	Load the pAE values between the reshaped residues and the rest of the protein,
	as well as the pLDDT values of the reshaped residues.
	"""

	# Load the LHLR structure
	pose_design, bb_remodeled_residues, bb_fixed_residues \
		= load_design(data_dict['%s_path'%model_type], bb_remodeled_residues=bb_remodeled_residues, design_info_file=design_info_file)

	bb_remodeled_residues_all = [i for lhl in bb_remodeled_residues for i in lhl]

	try:
		# Find pAE values between the bb_remodeled_residues and bb_fixed_residues (pAE is 0-indexed, while residues are 1-indexed)
		paes = [line.replace('[','').replace(']','').replace("'","").split() for line in data_dict[model_type+'_pae'].split('] [')]
		reshaped_paes = []
		## For now, treating all bb_remodeled_residues as one. In the future, could separate LHL1 and 2.
		for i in bb_remodeled_residues_all:
			for j in bb_fixed_residues:
				reshaped_paes.append(float(paes[i-1][j-1])) # (pAE is 0-indexed, while residues are 1-indexed)
	except:
		print(f"Failed to load reshaped pAEs. Setting to -999")
		reshaped_paes = [-999]

	try:
		# Find pLDDT values between the bb_remodeled_residues and bb_fixed_residues (pLDDT is 0-indexed, while residues are 1-indexed)
		plddts = data_dict[model_type+'_plddt'].replace('[','').replace(']','').replace("'","").split()
		reshaped_plddts = []
		## For now, treating all bb_remodeled_residues as one. In the future, could separate LHL1 and 2.
		for i in bb_remodeled_residues_all:
			reshaped_plddts.append(float(plddts[i-1])) # (pLDDT is 0-indexed, while residues are 1-indexed)
	except:
		print(f"Failed to load reshaped pLDDTs. Setting to -999")
		reshaped_plddts = [-999]

	return np.mean(reshaped_paes), np.mean(reshaped_plddts)


def find_design_info_file(args, data_dict):
	for design_info_folder in args.design_info_folders:
		if str(data_dict['name'])+'_design_info.json' in os.listdir(design_info_folder):
			return os.path.join(design_info_folder, str(data_dict['name'])+'_design_info.json')


##########
##########
##########


if __name__ == '__main__':

	# Parse command line arguments.
	parser = argparse.ArgumentParser(
		description="Calculate RMSDs between classic and fine-tuned AF2 predictions to their ground truth structures.  \
		Load and save a dataframe with the checkpointed models' predictions' RMSDs to the ground truth structures, their\
		confidence metrics, and their paths to the PDB files for their classic AF2, fine-tuned AF2, and ground truth\
		structures.",
		formatter_class=argparse.RawDescriptionHelpFormatter)

	parser.add_argument('--job_name', type=str, required=True,
			help='Name for the job which will be used in the output .png files.')
	parser.add_argument('--predictions', type=str, required=True,
			help='Path to a folder containing classic and fine-tuned AF2 predictions (.pdb files), as well as AF2 confidence scores (.npy files).')
	parser.add_argument('--ground_truth', nargs='*', type=str, required=True,
			help='A folder (or folders) containing PDB files of the ground truth structures (e.g., design models) of the corresponding\
			AF2 predictions.  The PDB headers in the ground truth files must also appear in the headers of the AF2 output files.')
	parser.add_argument('--model_names', nargs=2, type=str, required=True,
			help='The classic and fine-tuned AF2 model names (as they appear in the headers of the prediction output files).\
			The classic model name should be provided first, followed by the fine-tuned model name.')
	parser.add_argument('--outfolder_parent', type=str, required=True,
			help='Output files are written to outfolder_parent.')
	parser.add_argument('--reshaped_helix_rmsd', action='store_true',
			help='Calculate Reshaped Helix RMSD to Rosetta values for LUCS designs.')
	parser.add_argument('--reshaped_residue_confidence', action='store_true',
			help='Calculate the pAE values between the bb_remodeled_residues and non-bb_remodeled_residues and\
			pLDDT values of the bb_remodeled_residues (this residue set is found in the bb_remodeled_residues column\
			of the stability_data dataframe(s)). Uses the default pLDDT and pAE thresholds or the provided thresholds.')
	parser.add_argument('--design_info_folders', nargs='*', type=str, required=False, default=None,
			help="If using the --reshaped_helix_rmsd flag or the --reshaped_residue_confidence flag,\
			then a path to a folder (or folders) containing LUCS designs'\
			<design_id>_design_info.json files must be provided with this flag.")
	parser.add_argument('--verbose', action='store_true',
			help='Print extra print statements.')
	parser.add_argument('--test_stop', type=int, required=False, default=None,
			help='Number of test calculations to perform.')

	args = parser.parse_args()

	outfolder = os.path.join(args.outfolder_parent)
	os.makedirs(outfolder, exist_ok=True)

	gt_file_list = [os.path.join(gt_dir,d) for gt_dir in args.ground_truth for d in os.listdir(gt_dir)]

	if args.verbose:
		print('Length of ground truth file list:')
		print(len(gt_file_list))

	# Load a list of dictionaries, each containing ground truth names,
	# model names, and the paths to the models predicted pdbs.
	list_of_dicts = []
	af2_filelist = os.listdir(args.predictions)

	if args.verbose:
		print('Length of af2_filelist:')
		print(len(af2_filelist))

	prev_files = []
	file_idx = 0
	for af2_file in af2_filelist:

		if '.pdb' not in af2_file:
			continue

		# Skip examples for which there are no ground truth files.
		no_gt_file = True
		for gt_file in gt_file_list:
			# Check if the ground truth file exists by looking for the ground truth header
			# in the AF2 prediction file.
			if '_'+gt_file.split('/')[-1].replace('.pdb','')+'_' in af2_file.split('/')[-1]:
				no_gt_file = False
				if args.verbose:
					print('AF2 file, gt file pair')
					print(af2_file)
					print(gt_file)
				break

		# Ensure the current file hasn't already been run, and that there is a ground truth file for the example.
		if af2_file in prev_files or no_gt_file:
			if args.verbose:
				if no_gt_file:
					print(f'No GT file for {af2_file}')
			continue

		# Make a new dictionary for the current predicted example.
		new_dict = {}

		af2_file_split = af2_file.split('model_',1) # split at the first occurence

		for af2_f2 in af2_filelist:
			if af2_f2 in prev_files: # ensure the current file hasn't already been run
				continue

			# Find classic and fine-tuned AF2 model predictions for a single example.
			if '.pdb' in af2_f2 and af2_file_split[0] in af2_f2 and af2_file != af2_f2:

				if args.verbose:
					print('Paired AF2 predictions:')
					print(af2_file)
					print(af2_f2)

				# Ensure that classic AF2 is always model 1, and fine-tuned AF2 is always model 2.
				# if args.model_names[0] in af2_file and args.model_names[1] in af2_f2:
				# if any(e in af2_file for e in ['model_1_ptm','model_2_ptm','model_3_ptm','model_4_ptm','model_5_ptm']):
				if 'ft' in af2_f2 and 'ft' not in af2_file:
					model1_file = af2_file
					model2_file = af2_f2
				else:
					model2_file = af2_file
					model1_file = af2_f2

				if args.verbose:
					print('Model 1 file: %s'%model1_file)
					print('Model 2 file: %s'%model2_file)

				# new_dict['name'] = model1_file.split('_')[1] # name corresponds to the design name.
				new_dict['name'] = model1_file.split('_model')[0].replace(args.job_name+'_', '') # name corresponds to the design name.

				# For the Round 2 2LV8 (RO2) designs, their PDBs are named {design_id}.pdb, not {design_id}_{fold_name}.pdb
				if '2LV8' in new_dict['name']:
					new_dict['name'] = new_dict['name'].replace('_2LV8', '')

				new_dict['classic_path'] = os.path.join(args.predictions,model1_file)
				new_dict['ft_path'] = os.path.join(args.predictions,model2_file)
				new_dict['classic_name'] = args.model_names[0]
				new_dict['ft_name'] = args.model_names[1]

				# Load both models' confidence scores into the current example's dictionary.
				new_dict = load_confidence_scores(new_dict,args.predictions,
					model1_file,model2_file)

				# Add the example's dictionary to the list of dictionaries.
				list_of_dicts.append(new_dict)

				# Ensure the same examples aren't run twice.
				prev_files.append(af2_file)
				prev_files.append(af2_f2)

		file_idx += 1
		if args.test_stop != None:
			if file_idx > args.test_stop:
				break


	# Once all the predictions have been loaded into the list_of_dicts, add the ground truth
	# paths and names to each subdictionary in the list_of_dicts.
	for gt_file in gt_file_list:
		for data_dict in list_of_dicts:
			if '_'+gt_file.split('/')[-1].replace('.pdb','')+'_' in data_dict['classic_path'].split('/')[-1]:
				data_dict['ground_truth_path'] = gt_file
				data_dict['ground_truth_name'] = gt_file.split('/')[-1].replace('.pdb','')

	# Calculate the RMSDs between each example's ground truth structure and two predictions.
	for data_dict in list_of_dicts:
		# in case a ground truth file has been corrupted, adding a try/except clause
		# try:
		print('Calculating RMSD for target: %s' % data_dict['ground_truth_name'])
		parser = PDBParser(QUIET=True)
		
		try:
			gt_struct = parser.get_structure(data_dict['ground_truth_name'], data_dict['ground_truth_path'])
		except:
			print(f"Ground Truth structure {data_dict['ground_truth_path']} could not be loaded. Skipping this structure.")
			continue
		try:
			model1_struct = parser.get_structure(data_dict['classic_name'], data_dict['classic_path'])
		except:
			print(f"Classic AF2 predicted structure {data_dict['classic_path']} could not be loaded. Skipping this structure.")
			continue
		try:
			model2_struct = parser.get_structure(data_dict['ft_name'], data_dict['ft_path'])
		except:
			print(f"Fine-tuned AF2 predicted structure {data_dict['ft_path']} could not be loaded. Skipping this structure.")
			continue

		# Store structures with their model_types
		model_type_struct_dict = {'classic': model1_struct, 'ft': model2_struct}

		atom_types = ['N','CA','C']

		# range(1, len+1) because res_idxs are 1-indexed (PDB-numbering)
		# If the ground truth and prediction PDB files have the same numbering, then
		# defining only one set of align and rmsd residues works:
		# align_res_subset = list(range(1, len(list(gt_struct.get_residues()))+1))
		# rmsd_res_subset = align_res_subset

		# This assumes no chain breaks in the ground truth or predicted structures.

		# print(f"Ben Orr 10.30.24: Output from gt_struct.get_residues())[0].get_id()")
		# print(list(gt_struct.get_residues())[0].get_id()[1]) # .get_id() returns tuple (' ', 1, ' ')
		# print(list(gt_struct.get_residues())[-1].get_id()[1])

		# Get the chain the Ground Truth structure
		# This doesn't work, because the some ground truth files have multiple chains, and this selects the first chain
		# gt_chain = str(list(gt_struct.get_chains())[0]).split('=')[-1].replace('>','') # gt_struct.get_chains())[0] returns <Chain id=A>
		# print(f"Ben Orr 10.30.24: Ground Truth file chain: {gt_chain}")

		# Instead, use the CATH4.2 domain ID to get the chain
		# CATH4.2 domains: PDBIDDOMAINID (4 + 3)
		if len(data_dict['ground_truth_name']) == 7 and data_dict['ground_truth_name'][-1].isalpha(): # and data_dict['ground_truth_name'][4].isalpha(): # some chain IDs are numerical
			gt_chain_id = data_dict['ground_truth_name'][4]
		# De novo dataset 4.7.25: PDBID_CHAINID (5 + 1 + 1)
		elif len(data_dict['ground_truth_name']) == 6 and data_dict['ground_truth_name'][-1].isalpha():
			gt_chain_id = data_dict['ground_truth_name'][-1]
		else:
			print(f"Current structure ({data_dict['ground_truth_path']}) is not in CATH4.2 domain format. Setting Ground Truth Chain to 'A'")
			gt_chain_id = 'A'

		print(f"Ben Orr 10.30.24: Ground Truth file chain: {gt_chain_id}")

		gt_model = gt_struct[0]
		gt_chain = gt_model[gt_chain_id]
		model1_model = model1_struct[0]
		model1_chain = model1_model["A"]

		gt_align_res_subset = list(range(list(gt_chain.get_residues())[0].get_id()[1], list(gt_chain.get_residues())[-1].get_id()[1]+1))
		gt_rmsd_res_subset = gt_align_res_subset

		model_align_res_subset = list(range(list(model1_chain.get_residues())[0].get_id()[1], list(model1_chain.get_residues())[-1].get_id()[1]+1)) # This assumes that model1 and model2 have the same PDB numbering
		model_rmsd_res_subset = model_align_res_subset

		print(f"Ben Orr 10.30.24: gt and model align res subsets:")
		print(gt_rmsd_res_subset)
		print(model_rmsd_res_subset)

		model1_gt_rmsd = get_RMSD(gt_struct,
							model1_struct,
			 				atom_types, 
							gt_align_res_subset,
							gt_rmsd_res_subset,
							model_align_res_subset,
							model_rmsd_res_subset,
							ref_chain=gt_chain_id)
		data_dict['classic_gt_rmsd'] = model1_gt_rmsd

		model2_gt_rmsd = get_RMSD(gt_struct, 
							model2_struct,
							atom_types, 
							gt_align_res_subset,
							gt_rmsd_res_subset,
							model_align_res_subset,
							model_rmsd_res_subset,
							ref_chain=gt_chain_id)
		data_dict['ft_gt_rmsd'] = model2_gt_rmsd

		print('AF2-ground truth RMSD for %s: %.4f' % (data_dict['classic_name'], model1_gt_rmsd))
		print('AF2-ground truth RMSD for %s: %.4f' % (data_dict['ft_name'], model2_gt_rmsd))

		# Calculate Helix RMSDs between the ground truth structures and the two predictions.
		if args.reshaped_helix_rmsd:
			if args.design_info_folders == None:
				print('--design_info_folders must be provided if using the --reshaped_helix_rmsd option.')
			else:
				for model_type in model_type_struct_dict:
					# try:
					# helix_rmsd = calc_helix_RMSD(data_dict, args, model_type=model_type)

					model_struct = model_type_struct_dict[model_type]

					design_info_file = find_design_info_file(args, data_dict)

					model = list(model_struct.get_models())[0]
					chain = list(model.get_chains())[0]
					sequence = list(chain.get_residues())

					lhl_residues, helix_residues = get_lhl_residues(
						design_info_file = design_info_file, 
						res_start_idx = 1,
						seq_len = len(sequence))

					if args.verbose:
						print('Helix Residues:')
						print(helix_residues)
						print('LHL Residues:')
						print(lhl_residues)

					helix_idxs = [i for i in range(len(helix_residues))]
					bb_remodeled_residues = [i for lhl in lhl_residues for i in lhl]

					if args.verbose:
						print('helix_idxs:')
						print(helix_idxs)
						print('bb_remodeled_residues:')
						print(bb_remodeled_residues)

					# Calculate Helix RMSDs between the current AF2 prediction and FastDesign Struct
					# Second returned value is a list of tuples of helix start-stop indices.
					helix_rmsd, _ = calculate_bb_remodeled_region_rmsd(data_dict['ground_truth_path'],
							data_dict[f'{model_type}_path'],
							helix_idxs=helix_idxs,
							is_af2_design=True,
							atom_types=atom_types,
							ref_lhl_residues=lhl_residues,
							pose_lhl_residues=lhl_residues,
							ref_helix_residues=helix_residues,
							pose_helix_residues=helix_residues)

					data_dict[f'{model_type}_helix_rmsd'] = helix_rmsd
					print('Helix RMSD for %s to ground truth: %.3f'%(model_type, helix_rmsd))

					for helix_idx in helix_idxs:
						# Calculate Helix RMSDs between the current AF2 prediction and FastDesign Struct
						# Second returned value is a list of tuples of helix start-stop indices.
						helix_rmsd, _ = calculate_bb_remodeled_region_rmsd(data_dict['ground_truth_path'],
								data_dict[f'{model_type}_path'],
								helix_idxs=[helix_idx],
								is_af2_design=True,
								atom_types=atom_types,
								ref_lhl_residues=lhl_residues,
								pose_lhl_residues=lhl_residues,
								ref_helix_residues=helix_residues,
								pose_helix_residues=helix_residues)

						data_dict[f'{model_type}_helix{helix_idx}_rmsd'] = helix_rmsd
						print(f'Helix {helix_idx} RMSD for {model_type} to ground truth: {helix_rmsd:.4f}')

					# except:
					# 	print('Helix RMSD for %s could not be calculated (perhaps no helical residues found).\n\
					# 		\nSetting Helix RMSD value to None'%(model_type))
					# 	data_dict['%s_helix_rmsd'%model_type] = None

		# Calculate the reshaped residues confidence for the two predictions.
		if args.reshaped_residue_confidence:
			if args.design_info_folders == None:
				print('--design_info_folders must be provided if using the --reshaped_residue_confidence option.')
			else:
				for model_type in model_type_struct_dict:
					# try:
					reshaped_pae, reshaped_plddt = get_reshaped_confidence(data_dict, args, model_type=model_type, bb_remodeled_residues=bb_remodeled_residues, design_info_file=design_info_file)
					data_dict['%s_reshaped_pae'%model_type] = reshaped_pae
					data_dict['%s_reshaped_plddt'%model_type] = reshaped_plddt
					print('Reshaped Residue pAE and pLDDT for %s: %.3f, %.3f'%(model_type, reshaped_pae, reshaped_plddt))

					# except:
					# 	print('Reshaped Residue pAE and pLDDT for %s could not be calculated (perhaps no design_info.json file found).\n\
					# 		\nSetting Reshaped Residue pAE and pLDDT values to None'%(model_type))
					# 	data_dict['%s_reshaped_pae'%model_type] = None
					# 	data_dict['%s_reshaped_plddt'%model_type] = None
		# except:
		# 	print('Skipping design %s: Ground truth file likely not found.'% data_dict['ground_truth_name'])
		# 	data_dict['classic_gt_rmsd'] = None
		# 	data_dict['ft_gt_rmsd'] = None

	# print('Final list of dictionaries:')
	# print(list_of_dicts)

	df = pd.DataFrame(list_of_dicts)
	df_csv = f'{args.job_name}.csv'
	df.to_csv(os.path.join(outfolder, df_csv))
	print('Saved %s'%os.path.join(outfolder, df_csv))

	# compare_rmsds(list_of_dicts, args.job_name, outfolder)


