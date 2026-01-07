"""
Provides utility functions for analyzing LUCS designs and AlphaFold predictions.

Author: Ben Orr
Date: 8.21.23

This module provides comprehensive tools for analyzing Loop-Helix-Loop Unit
Combinatorial Sampling (LUCS) designs, with particular focus on:

- Calculating degree of reshaping between designed and predicted structures
- Computing AlphaFold confidence metrics (pLDDT scores)
- Structural alignment and RMSD calculations
- Helix geometry analysis (angles, displacements, directions)
- Secondary structure element analysis (helices, sheets)
- TM-score calculations for structural similarity
- Structure file I/O and format conversions (PDB, CIF)
- Integration with PyRosetta for structural manipulation

The primary use case is evaluating how well AlphaFold predictions match the
intended LUCS designs by quantifying structural deviations in backbone-remodeled
regions and secondary structure elements.

Dependencies
------------
- BioPython: PDB/CIF parsing, DSSP secondary structure
- PyRosetta: Structure manipulation, alignment, RMSD calculations
- tmtools: TM-score calculations
"""

import argparse

# For Reshaped Helix Angle and Displacement
from Bio.PDB.DSSP import DSSP
from Bio.PDB import *

# For conversion between CIF and PDB format
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBIO

import os
import sys
import re
from glob import glob
import json
import gzip
import shutil
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pickle as pkl
import pandas as pd

# For helix RMSD calculations
import pyrosetta
from pyrosetta import rosetta

# For alignment and RMSD calculations
# from pyrosetta import init
import pyrosetta.distributed.dask
from pyrosetta import pose_from_file
from pyrosetta.rosetta.core.simple_metrics.metrics import RMSDMetric
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector, OrResidueSelector
from pyrosetta.rosetta.protocols.fold_from_loops.movers import AlignByResidueSelectorMover

# for Steph's Helix RMSD functions:
from pyrosetta.rosetta.protocols.moves import DsspMover
from pyrosetta.rosetta.core.scoring import *
from pyrosetta.rosetta.core import *
from pyrosetta import pose_from_pdb

# init pyrosetta
# init(options='-mute all')

# tmtools for calculating TM score
from tmtools.io import get_structure, get_residue_data
from tmtools.testing import get_pdb_path
from tmtools import tm_align

def find_lucs_dir(args: argparse.Namespace, design_id: str) -> str:
	"""
	Find the LUCS directory for a given design ID.

	Searches through parent directories and subdirectories to locate the LUCS
	design directory. Supports both flat and nested subdirectory structures
	depending on the args.lucs_data_subdir_split flag.

	Parameters
	----------
	args : argparse.Namespace
		Command-line arguments containing:
		- lucs_data : list of str
			Parent directories to search
		- lucs_data_subdir_split : bool
			If True, search in nested subdirectories (parent/subdir/design_id)
			If False, search in flat subdirectories (parent/design_id)
	design_id : str
		The design identifier to search for

	Returns
	-------
	str or None
		Full path to the LUCS directory if found, None otherwise

	Notes
	-----
	Prints a warning message if no LUCS directory is found.
	"""
	found = False
	if args.lucs_data_subdir_split == True:
		for parent_dir in args.lucs_data:
			for subdir in os.listdir(parent_dir):
				# if subdir.isdigit() == False:
				if os.path.isdir(os.path.join(parent_dir, subdir)) == False:
					continue
				for s2 in os.listdir(os.path.join(parent_dir, subdir)):
					if str(s2) == str(design_id):
						lucs_dir = os.path.join(parent_dir, subdir, s2)
						return lucs_dir
	else:
		for parent_dir in args.lucs_data:
			for subdir in os.listdir(parent_dir):
				# if subdir.isdigit() == False:
				if os.path.isdir(os.path.join(parent_dir, subdir)) == False:
					continue
				if str(subdir) == str(design_id):
					lucs_dir = os.path.join(parent_dir, subdir)
					return lucs_dir

	if found==False:
		print('\nWarning: No LUCS directory found!\n')
		return None

def find_original_lucs_dir(args: argparse.Namespace, design_id: str) -> str:
	"""
	Find the original LUCS directory or file for a given design ID.

	Searches for the original LUCS backbone in various directory
	configurations, including compatible LHL formats and nested subdirectories.

	Parameters
	----------
	args : argparse.Namespace
		Command-line arguments containing:
		- original_lucs_data : str
			Parent directory containing original LUCS data
		- original_LUCS_data_compatible_LHLs : bool
			If True, uses model_{design_id}.pdb naming convention
		- original_lucs_data_subdir_split : bool
			If True, search in nested subdirectories
	design_id : str
		The design identifier to search for

	Returns
	-------
	str or None
		Full path to the original LUCS directory/file if found, None otherwise

	Notes
	-----
	Prints a warning message if no LUCS directory is found.
	"""
	found = False
	if args.original_LUCS_data_compatible_LHLs:
		lucs_path = os.path.join(args.original_lucs_data, f'model_{design_id}.pdb')
		return lucs_path
	elif args.original_lucs_data_subdir_split:
		for subdir in os.listdir(args.original_lucs_data):
			# if subdir.isdigit() == False:
			if os.path.isdir(os.path.join(args.original_lucs_data, subdir)) == False:
				continue
			for s2 in os.listdir(os.path.join(args.original_lucs_data, subdir)):
				if str(s2) == str(design_id):
					lucs_dir = os.path.join(args.original_lucs_data, subdir, s2)
					return lucs_dir
	else:
		for subdir in os.listdir(args.original_lucs_data):
			# if subdir.isdigit() == False:
			if os.path.isdir(os.path.join(args.original_lucs_data, subdir)) == False:
				continue
			if str(subdir) == str(design_id):
				lucs_dir = os.path.join(args.original_lucs_data, subdir)
				return lucs_dir

	if found==False:
		print('\nWarning: No LUCS directory found!\n')
		return None

def pdb_to_seq(pdb_path: str) -> tuple[str, str]:
	"""
	Extract the protein sequence from a pdb file.

	Parses ATOM records from a pdb file and converts three-letter amino acid
	codes to single-letter sequence. Returns the sequence and chain ID.

	Parameters
	----------
	pdb_path : str
		Path to the pdb file

	Returns
	-------
	tuple[str, str]
		A tuple containing:
		- seq : str
			The protein sequence in single-letter amino acid code
		- chain_id : str
			The chain identifier (assumes single chain)

	Notes
	-----
	- Currently only handles pdb format, not CIF files (TODO)
	- Assumes the pdb file contains only one chain
	- Reads all ATOM records regardless of chain ID
	- Handles non-standard residue numbering by tracking unique residue numbers

	TODO
	----
	Handle .cif files
	"""
	aa_dict = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
		'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
		'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
		'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
	
	# chain = 'A'

	resnums = []
	seq = ''
	chain_ids = []
	with open(pdb_path, 'r') as f:
		for line in f.readlines():
			if len(line.split()) > 0:
				if line.startswith('ATOM'): # and line[22] == chain: # Commenting out for cases non-chain A
					if line[21] not in chain_ids:
						chain_ids.append(line[21])
					if int(line[22:26]) in resnums:
						continue
					else:
						seq += aa_dict[line[17:20]]
						resnums.append(int(line[22:26]))

	f.close()
	return seq, chain_ids[0] # Assumes that the PDB contains only one chain

def load_af2_structs(af_dir: str, header: str = 'unrelaxed', suffix: str = '_ptm.pdb') -> tuple:
	"""
	Load the 5 AlphaFold2 structures from a directory.

	Loads all 5 AF2 model predictions as PyRosetta Pose objects from the
	specified directory using standard AF2 naming conventions.

	Parameters
	----------
	af_dir : str
		Directory containing AlphaFold2 prediction files
	header : str, optional
		Filename prefix for AF2 models (default: 'unrelaxed')
	suffix : str, optional
		Filename suffix for AF2 models (default: '_ptm.pdb')

	Returns
	-------
	tuple
		A tuple containing:
		- af_struct_1 through af_struct_5 : pyrosetta.Pose
			PyRosetta Pose objects for models 1-5
		- af_paths : list of str
			List of file paths to the 5 models

	Notes
	-----
	Expected filename format: {header}_model_{i}{suffix} where i is 1-5
	"""
	af_struct_1 = pose_from_file(
		os.path.join(af_dir,f'{header}_model_1{suffix}'))
	af_struct_2 = pose_from_file(
		os.path.join(af_dir,f'{header}_model_2{suffix}'))
	af_struct_3 = pose_from_file(
		os.path.join(af_dir,f'{header}_model_3{suffix}'))
	af_struct_4 = pose_from_file(
		os.path.join(af_dir,f'{header}_model_4{suffix}'))
	af_struct_5 = pose_from_file(
		os.path.join(af_dir,f'{header}_model_5{suffix}'))
	af_paths = [os.path.join(af_dir,f'{header}_model_{i}{suffix}') for i in range(1,6)]
	return af_struct_1, af_struct_2, af_struct_3, af_struct_4, af_struct_5, af_paths

def load_af3_structs(af_dir: str, design_id: str = '') -> tuple:
	"""
	Load the 5 AlphaFold3 structures from AlphaFold Server output.

	Loads AF3 model predictions as PyRosetta Pose objects, converting CIF
	files to PDB format if necessary. Works with the output format from the
	AlphaFold Server.

	Parameters
	----------
	af_dir : str
		Directory containing AlphaFold3 prediction files
	design_id : str, optional
		Design identifier used in filenames (default: '')

	Returns
	-------
	tuple
		A tuple containing:
		- af_struct_1 through af_struct_5 : pyrosetta.Pose
			PyRosetta Pose objects for models 0-4
		- af_pdb_paths : list of str
			List of pdb file paths for the 5 models

	Notes
	-----
	- Automatically converts CIF files to pdb format if pdb files don't exist
	- Expected CIF filename format: {design_id}_model_{i}.cif where i is 0-4
	- Generated pdb filename format: {design_id}_model_{i}.pdb where i is 0-4
	- Prints conversion messages during CIF to PDB conversion

	Works for AF3 predictions in the format output by the AlphaFold Server
	"""

	af_cif_paths = [os.path.join(af_dir,f'{design_id}_model_{i}.cif') for i in range(5)]

	af_pdb_paths = [os.path.join(af_dir,f'{design_id}_model_{i}.pdb') for i in range(5)]

	for af_pdb_path, af_cif_path in zip(af_pdb_paths, af_cif_paths):
		if os.path.exists(af_pdb_path):
			print(f'pdb file exists: {af_pdb_path}')
		else:
			# Read file
			parser = MMCIFParser()
			structure = parser.get_structure(af_cif_path.split('.')[0], af_cif_path)
			# Write pdb
			io = PDBIO()
			io.set_structure(structure)
			io.save(af_pdb_path)
			print(f'Converted {af_cif_path} to {af_pdb_path}')

	[af_struct_1, af_struct_2, af_struct_3, af_struct_4, af_struct_5] = \
		[pyrosetta.rosetta.core.import_pose.pose_from_file(af_path) for af_path in af_pdb_paths]

	return af_struct_1, af_struct_2, af_struct_3, af_struct_4, af_struct_5, af_pdb_paths

def load_af3_linux_structs(args: argparse.Namespace, af_dir: str, design_id: str = '') -> tuple:
	"""
	Load the 5 AlphaFold3 structures from Linux installation output.

	Loads AF3 model predictions as PyRosetta Pose objects from the directory
	structure produced by UCSF Wynton's AlphaFold3 Linux installation. Converts
	CIF files to PDB format if necessary.

	Parameters
	----------
	args : argparse.Namespace
		Command-line arguments containing:
		- is_af3_linux : bool
			Flag indicating AF3 Linux installation format
	af_dir : str
		Directory containing AlphaFold3 prediction subdirectories
	design_id : str, optional
		Design identifier used in output filenames (default: '')

	Returns
	-------
	tuple
		A tuple containing:
		- af_struct_1 through af_struct_5 : pyrosetta.Pose
			PyRosetta Pose objects for models 0-4
		- af_pdb_paths : list of str
			List of pdb file paths for the 5 models

	Notes
	-----
	- Automatically converts CIF files to pdb format if pdb files don't exist
	- Expected directory structure: seed-1_sample-{n}/model.cif for n=0-4
	- Generated pdb filename format: {design_id}_model_{i}.pdb where i is 0-4
	- Currently hard-coded to use random seed 1
	- Prints conversion messages during CIF to PDB conversion

	Works for AF3 predictions in the format output by UCSF Wynton's AlphaFold3 installation
	"""

	# Hard-coding in random seed 1
	af_cif_paths = []
	for i, d in enumerate([f"seed-1_sample-{n}" for n in range(5)]):
		af_cif_paths.append((i, os.path.join(af_dir, d, 'model.cif')))

	# Sort by sample number
	af_cif_paths = [path for _, path in sorted(af_cif_paths, key=lambda x: x[0])]

	af_pdb_paths = [os.path.join(af_dir,f'{design_id}_model_{i}.pdb') for i in range(5)]

	for af_pdb_path, af_cif_path in zip(af_pdb_paths, af_cif_paths):
		if os.path.exists(af_pdb_path):
			print(f'pdb file exists: {af_pdb_path}')
		else:
			# Read file
			parser = MMCIFParser()
			if args.is_af3_linux:
				structure = parser.get_structure(af_cif_path.split('.')[0], af_cif_path)
			else:
				structure = parser.get_structure(af_cif_path.split('.')[0], af_cif_path)
			# Write PDB
			io = PDBIO()
			io.set_structure(structure)
			io.save(af_pdb_path)
			print(f'Converted {af_cif_path} to {af_pdb_path}')

	[af_struct_1, af_struct_2, af_struct_3, af_struct_4, af_struct_5] = \
		[pyrosetta.rosetta.core.import_pose.pose_from_file(af_path) for af_path in af_pdb_paths]

	return af_struct_1, af_struct_2, af_struct_3, af_struct_4, af_struct_5, af_pdb_paths

def load_colabfold_structs(args: argparse.Namespace, af_dir: str, design_id: str) -> tuple:
	"""
	Load the 5 AlphaFold2 structures from a ColabFold output folder.

	Loads ColabFold AF2 model predictions as PyRosetta Pose objects. Identifies
	structures by design ID prefix and model number suffix.

	Parameters
	----------
	args : argparse.Namespace
		Command-line arguments containing:
		- verbose : bool
			If True, print detailed file information
	af_dir : str
		Directory containing ColabFold prediction files
	design_id : str
		Design identifier used as filename prefix

	Returns
	-------
	tuple
		A tuple containing:
		- af_struct_1 through af_struct_5 : pyrosetta.Pose
			PyRosetta Pose objects for models 1-5
		- af_paths : list of str
			List of file paths to the 5 models

	Notes
	-----
	- Provide a design ID that uniquely identifies exactly 5 AF2 predictions
	- For example, use '123_mpnnSeq1' not '123' if '123_mpnnSeq2' also exists
	- Files are identified by design_id prefix and model number suffix (1-5)
	- Verbose mode prints current directory and found pdb files

	Get the 5 structures by their design ID.

	Provide a design ID that is found only its 5 AF2 predictions. For example, provide
	('123_mpnnSeq1' and not '123' as the design ID, if there also exists '123_mpnnSeq2').
	"""
	pdb_files = []
	for file in os.listdir(af_dir):
		if file.startswith(design_id) and file.split('.')[-1] == 'pdb':
			pdb_files.append(file)

	if args.verbose:
		print(f"Current af_dir: {af_dir}")
		print(f'pdb files found: {pdb_files}')

	af_struct_1 = pose_from_file(
		os.path.join(af_dir, [f for f in pdb_files if f.split('.')[0].split('_')[-1] == '1'][0]))
	af_struct_2 = pose_from_file(
		os.path.join(af_dir, [f for f in pdb_files if f.split('.')[0].split('_')[-1] == '2'][0]))
	af_struct_3 = pose_from_file(
		os.path.join(af_dir, [f for f in pdb_files if f.split('.')[0].split('_')[-1] == '3'][0]))
	af_struct_4 = pose_from_file(
		os.path.join(af_dir, [f for f in pdb_files if f.split('.')[0].split('_')[-1] == '4'][0]))
	af_struct_5 = pose_from_file(
		os.path.join(af_dir, [f for f in pdb_files if f.split('.')[0].split('_')[-1] == '5'][0]))
	af_paths = [os.path.join(af_dir, [f for f in pdb_files if f.split('.')[0].split('_')[-1] == str(i)][0]) for i in range(1,6)]
	return af_struct_1, af_struct_2, af_struct_3, af_struct_4, af_struct_5, af_paths

def get_colabfold_struct_paths(args: argparse.Namespace, af_dir: str, design_id: str) -> list[str]:
	"""
	Get file paths for ColabFold structures in model order.

	Returns a list of full file paths to the 5 ColabFold prediction files,
	ordered by AF2 model number (1-5).

	Parameters
	----------
	args : argparse.Namespace
		Command-line arguments containing:
		- verbose : bool
			If True, print found prediction files
	af_dir : str
		Directory containing ColabFold prediction files
	design_id : str
		Design identifier used as filename prefix

	Returns
	-------
	list of str
		List of 5 file paths ordered by AF2 model number (1-5)

	Notes
	-----
	- Provide a design ID that uniquely identifies exactly 5 AF2 predictions
	- For example, use '123_mpnnSeq1' not '123' if '123_mpnnSeq2' also exists
	- Verbose mode prints list of found ColabFold predictions

	Return a list (in order of AF2 model) of Colabfold structure filenames.

	Provide a design ID that is found only its 5 AF2 predictions. For example, provide
	('123_mpnnSeq1' and not '123' as the design ID, if there also exists '123_mpnnSeq2').
	"""
	pdb_files = []
	for file in os.listdir(af_dir):
		if file.startswith(design_id) and file.split('.')[-1] == 'pdb':
			pdb_files.append(file)

	if args.verbose:
		print(f'ColabFold predictions found for {design_id}:')
		print(pdb_files)

	path1 = os.path.join(af_dir, [f for f in pdb_files if f.split('.')[0].split('_')[-1] == '1'][0])
	path2 = os.path.join(af_dir, [f for f in pdb_files if f.split('.')[0].split('_')[-1] == '2'][0])
	path3 = os.path.join(af_dir, [f for f in pdb_files if f.split('.')[0].split('_')[-1] == '3'][0])
	path4 = os.path.join(af_dir, [f for f in pdb_files if f.split('.')[0].split('_')[-1] == '4'][0])
	path5 = os.path.join(af_dir, [f for f in pdb_files if f.split('.')[0].split('_')[-1] == '5'][0])

	return [path1, path2, path3, path4, path5]

def load_af_plddts(af_dir: str, args: argparse.Namespace, design_id: str = '') -> list:
	"""
	Load pLDDT confidence scores from AlphaFold predictions.

	Loads pLDDT (predicted Local Distance Difference Test) scores for all 5
	AlphaFold models. Handles different AF2/AF3 output formats based on args.

	Parameters
	----------
	af_dir : str
		Directory containing AlphaFold prediction files
	args : argparse.Namespace
		Command-line arguments containing:
		- is_af3_linux : bool
			If True, load from AF3 Linux installation format
		- is_af3 : bool
			If True, load from AF3 Server format; if False, load AF2 format
	design_id : str, optional
		Design identifier used in filenames for AF3 Server format (default: '')

	Returns
	-------
	list
		List of 5 pLDDT arrays (one per model), where each array contains
		per-residue confidence scores

	Notes
	-----
	- AF3 Linux format: Reads from seed-1_sample-{n}/confidences.json
	- AF2 format: Reads from result_model_{i}_ptm.pkl files
	- AF3 Server format: Reads from {design_id}_full_data_{i}.json files
	- pLDDT scores range from 0-100, with higher values indicating higher confidence

	See Also
	--------
	load_colabfold_plddts : Load pLDDT scores from ColabFold output
	"""

	if args.is_af3_linux: # AF3 outputs from AlphaFold3 linux installation

		# Hard-coding in random seed 1
		results_files = []
		for i, d in enumerate([f"seed-1_sample-{n}" for n in range(5)]):
			results_files.append((i, os.path.join(af_dir, d, 'confidences.json')))

		# Sort by sample number
		results_files = [path for _, path in sorted(results_files, key=lambda x: x[0])]

		results_plddts = []
		for result_file in results_files:
			with open(os.path.join(af_dir,result_file), 'rb') as f:
				result = json.load(f)
			results_plddts.append(result['atom_plddts'])
		return results_plddts
	
	elif not args.is_af3:
		# Load the results_model_<num>.pkl files into dictionaries
		results_files = [f'result_model_{i}_ptm.pkl' for i in range(1,6)]
		# result is a dictionary
		results_plddts = []
		for result_file in results_files:
			with open(os.path.join(af_dir,result_file), 'rb') as f:
				result = pkl.load(f)
			results_plddts.append(result['plddt'])
		return results_plddts

	else: # AF3 outputs from AlphaFold Server
		# Don't need {args.af_prefix} here, as af_design_id already contains this
		results_files = [f'{design_id}_full_data_{i}.json' for i in range(5)]
		results_plddts = []
		for result_file in results_files:
			with open(os.path.join(af_dir,result_file), 'rb') as f:
				result = json.load(f)
			results_plddts.append(result['atom_plddts'])
		return results_plddts


def load_colabfold_plddts(af_dir: str, design_id: str) -> list:
	"""
	Load pLDDT confidence scores from ColabFold predictions.

	Loads pLDDT scores from ColabFold score json files and returns them
	ordered by AF2 model number (1-5).

	Parameters
	----------
	af_dir : str
		Directory containing ColabFold prediction files
	design_id : str
		Design identifier used as filename prefix

	Returns
	-------
	list
		List of 5 pLDDT arrays (one per model), ordered by model number,
		where each array contains per-residue confidence scores

	Notes
	-----
	- Expected filename format: {design_id}_unrelaxed_rank_?_model_?_scores.json
	- Files are sorted by model number (1-5) before extracting pLDDT scores
	- pLDDT scores range from 0-100, with higher values indicating higher confidence

	Load the <design_id>_unrelaxed_rank_?_model_?_scores.json files into
	dictionaries
	"""
	files = []
	for file in os.listdir(af_dir):
		if design_id in file and file.split('.')[0].split('_')[-1] == 'scores' \
			and file.split('.')[-1] == 'json':
			files.append(file)

	# Sort the files so they're in the order of AF2 models 1-5
	sorted_files = []
	for i in range(1,6):
		for file in files:
			if file.split('.')[0].split('_')[-2] == str(i):
				sorted_files.append(file)

	results_plddts = []
	for file in sorted_files:
		with open(os.path.join(af_dir,file), 'rb') as f:
			result = json.load(f)
		results_plddts.append(result['plddt'])
	return results_plddts

def get_lhl_plddt_values(results_plddts: list, lhl_residues: list) -> tuple[np.ndarray, np.ndarray]:
	"""
	Extract pLDDT values for LHL regions and non-backbone-remodeled regions.

	Subsets pLDDT confidence scores into LHL (Loop-Helix-Loop) regions and
	remaining non-backbone-remodeled regions for all models.

	Parameters
	----------
	results_plddts : list
		List of pLDDT arrays, one per model (length = num_models)
	lhl_residues : list of list of int
		List of residue lists defining LHL regions, where each inner list
		contains 1-indexed residue numbers for one LHL unit

	Returns
	-------
	tuple of np.ndarray
		A tuple containing:
		- all_lhl_plddts : np.ndarray (object dtype)
			Shape: (num_models, num_lhls, variable_num_res)
			pLDDT values for each LHL region in each model
		- all_non_bb_rem_plddts : np.ndarray (object dtype)
			Shape: (num_models, variable_num_res)
			pLDDT values for non-LHL residues in each model

	Notes
	-----
	- Returns a num_models x num_lhls x num_res structure for LHL plddts
	- Residue indices in lhl_residues are 1-indexed
	- Non-backbone-remodeled regions include all residues not in any LHL

	Returns a num_models (len(results_plddts)) x num_lhls x num_res list of lists
	Get a subset of plddt values
	Load the sub_dict with lists of lists of plddt values.
	"""
	all_lhl_plddts = []
	all_non_bb_rem_plddts = []
	for plddts in results_plddts: # num_models
		all_lhl_plddts.append([])
		non_bb_rem_plddts = []
		for res_list in lhl_residues: # num_lhls
			lhl_plddts = []
			for i in range(len(plddts)):
				if i+1 in res_list: # num_res
					lhl_plddts.append(float(plddts[i]))
			all_lhl_plddts[-1].append(lhl_plddts)

		all_lhl_indices = [i+1 for res_list in lhl_residues for i in res_list]
		for i in range(len(plddts)):
			if i not in all_lhl_indices:
				non_bb_rem_plddts.append(float(plddts[i]))
		all_non_bb_rem_plddts.append(non_bb_rem_plddts)

	return np.array(all_lhl_plddts, dtype=object), np.array(all_non_bb_rem_plddts, dtype=object)

def load_plddts_into_subdict(sub_dict: dict, results_plddts: list,
                              all_lhl_plddts: np.ndarray,
                              all_non_bb_rem_plddts: np.ndarray) -> dict:
	"""
	Populate dictionary with pLDDT statistics for all AF models.

	Computes and stores 4 pLDDT summary statistics for each of the 5 AlphaFold
	models: overall mean, LHL mean, non-backbone-remodeled mean, and full array.

	Parameters
	----------
	sub_dict : dict
		Dictionary to populate with pLDDT values (modified in place)
	results_plddts : list
		List of 5 pLDDT arrays, one per AF model
	all_lhl_plddts : np.ndarray
		Array of shape (5, num_lhls, variable_num_res) containing pLDDT
		values for LHL regions
	all_non_bb_rem_plddts : np.ndarray
		Array of shape (5, variable_num_res) containing pLDDT values for
		non-backbone-remodeled regions

	Returns
	-------
	dict
		The input sub_dict populated with pLDDT statistics for each model:
		- '{model}_mean_plddt' : float
			Mean pLDDT across all residues
		- '{model}_mean_lhl_plddt' : float
			Mean pLDDT across LHL regions
		- '{model}_mean_non_bb_rem_plddt' : float
			Mean pLDDT across non-backbone-remodeled regions
		- '{model}_plddts' : list
			Full pLDDT array for all residues

	Notes
	-----
	- Modifies sub_dict in place and also returns it
	- model names are 'af_model1' through 'af_model5'

	Load the sub dict with 4 values for each of the 5 AF2/AF3 models.
	"""
	af_models = ['af_model1','af_model2','af_model3','af_model4','af_model5']
	for i in range(len(af_models)):
		af_model = af_models[i]
		sub_dict['ave_plddt_'+af_model] = np.mean(results_plddts[i])

		for j in range(len(all_lhl_plddts[i])):
			sub_dict[f'ave_lhl{j}_plddt_'+af_model] = np.mean(all_lhl_plddts[i][j])

		sub_dict['ave_bb_rem_plddt_'+af_model] = np.mean([val for lhl_plddts in all_lhl_plddts[i] for val in lhl_plddts]) # This should be all_lhl_plddts[i - res_start_idx] !
		sub_dict['ave_non_bb_rem_plddt_'+af_model] = np.mean(all_non_bb_rem_plddts[i])
		sub_dict['plddts_'+af_model] = ' '.join([str(i) for i in results_plddts[i]])

		for j in range(len(all_lhl_plddts[i])):
			sub_dict[f'lhl{j}_plddts_'+af_model] = ' '.join([str(r) for r in all_lhl_plddts[i][j]])

		sub_dict['bb_rem_plddts_'+af_model] = ' '.join([str(i) for lhl_plddts in all_lhl_plddts[i] for i in lhl_plddts])
		sub_dict['non_bb_rem_plddts_'+af_model] = ' '.join([str(i) for i in all_non_bb_rem_plddts[i]])

	sub_dict['ave_plddt_all_5_af_models'] = np.mean(np.array(results_plddts).flatten())

	return sub_dict

def get_biopython_atoms_with_ins_pts(struct, res_start_idx: int, atom_types: list[str],
                                      ins_pt1_start: int, ins_pt1_stop: int,
                                      ins_pt2_start: int, ins_pt2_stop: int) -> list:
	"""
	Extract BioPython atoms from non-backbone-remodeled regions.

	Returns BioPython atom objects for specified atom types from residues
	outside two insertion point (backbone-remodeled) regions.

	Parameters
	----------
	struct : Bio.PDB.Structure
		BioPython structure object
	res_start_idx : int
		Starting residue index (in case PDB numbering doesn't begin at 1)
	atom_types : list of str
		List of atom types to extract (e.g., ['CA'], ['N', 'CA', 'C'])
	ins_pt1_start : int
		Start index of first insertion point region
	ins_pt1_stop : int
		Stop index of first insertion point region (inclusive)
	ins_pt2_start : int
		Start index of second insertion point region
	ins_pt2_stop : int
		Stop index of second insertion point region (inclusive)

	Returns
	-------
	list
		List of BioPython Atom objects from non-backbone-remodeled regions

	Notes
	-----
	- Assumes chain A
	- Insertion point indices are adjusted by res_start_idx
	- Atoms are extracted only from residues outside both insertion regions

	Return biopython atoms within the 2 insertion point ranges from the
	provided biopython structure object.

	Residue start index is in case a pdb file's numbering doesn't begin on 1.
	"""
	ins_pt1_start+=res_start_idx-1
	ins_pt1_stop+=res_start_idx-1
	ins_pt2_start+=res_start_idx-1
	ins_pt2_stop+=res_start_idx-1

	ss_model = struct[0]
	ss_chain = ss_model["A"]

	ss_non_bb_remodeled_residues = []
	ss_bb_remodeled_residues = []

	ss_bb_remodeled_residues = list(range(ins_pt1_start,ins_pt1_stop)) + list(range(ins_pt2_start, ins_pt2_stop))
	ss_bb_remodeled_residues_lhl1 = list(range(ins_pt1_start,ins_pt1_stop))
	ss_bb_remodeled_residues_lhl2 = list(range(ins_pt2_start, ins_pt2_stop))

	for res_idx in range(res_start_idx, len(list(struct.get_residues()))+res_start_idx):
		if res_idx not in ss_bb_remodeled_residues:
			ss_non_bb_remodeled_residues.append(ss_chain[res_idx])

	ss_atoms = []
	for res in ss_non_bb_remodeled_residues:
			for atom_type in atom_types:
				# Get just the CA atoms with res['CA']
				ss_atoms.append(res[atom_type])
	return ss_atoms

## Writing my own function to calculate RMSD
def calc_RMSD(ref_coords: np.ndarray, coords: np.ndarray) -> float:
	"""
	Calculate Root Mean Square Deviation between two sets of coordinates.

	Computes RMSD as the square root of the mean squared Euclidean distance
	between corresponding points in two coordinate arrays.

	Parameters
	----------
	ref_coords : np.ndarray
		Reference coordinates, shape (num_points, num_dims)
	coords : np.ndarray
		Comparison coordinates, shape (num_points, num_dims)

	Returns
	-------
	float
		Root Mean Square Deviation between the two coordinate sets

	Notes
	-----
	- Typically used with 3D coordinates (num_dims=3)
	- Arrays must have same shape
	- No alignment is performed; points are compared in given order

	Takes num_points x num_dims (num_dims=3 for 3d coords) arrays, calculates the
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

def find_avg_helix_direction(resis: list, resis_idxs: list) -> np.ndarray:
	"""
	Calculate average helix direction vector from consecutive residue triplets.

	Computes helix direction from all valid consecutive triplets of residues
	and returns their mean direction vector.

	Parameters
	----------
	resis : list of dict
		List of residue dictionaries, each containing keys 'N', 'CA', 'C'
		with NumPy array coordinates as values
	resis_idxs : list of int
		List of residue indices corresponding to resis

	Returns
	-------
	np.ndarray
		Average helix direction vector (3D)

	Notes
	-----
	- Only uses consecutive triplets (res_i, res_i+1, res_i+2)
	- Each triplet must have contiguous residue indices
	- Direction is computed from peptide bond frame rotations

	Pass every set of 3 residues to the helix_direction function and return
	the average helix direction.
	Resis, currently, is a list of resi dictionaries. I can make my own
	resi dictionaries in this format, or change the functions to fit my current
	residue object structures.
	"""
	direction_vectors = []
	for i in range(0, len(resis) - 2):
		# Ensure triplets of residues are continuous
		if resis_idxs[i] + 1 == resis_idxs[i+1] and \
				resis_idxs[i+1] + 1 == resis_idxs[i+2]:
			direction_vectors.append(helix_direction(
				resis[i], resis[i+1], resis[i+2]
				))
	return np.array(direction_vectors).mean(axis=0)

def helix_direction(res1: dict, res2: dict, res3: dict) -> np.ndarray:
	"""
	Calculate helix direction from three consecutive residues.

	Computes the helix direction vector by analyzing the rotation between
	two consecutive peptide bond frames.

	Parameters
	----------
	res1 : dict
		First residue dictionary with keys 'N', 'CA', 'C' mapping to
		NumPy coordinate arrays
	res2 : dict
		Second residue dictionary with keys 'N', 'CA', 'C' mapping to
		NumPy coordinate arrays
	res3 : dict
		Third residue dictionary with keys 'N', 'CA', 'C' mapping to
		NumPy coordinate arrays

	Returns
	-------
	np.ndarray
		Helix direction vector (3D)

	Notes
	-----
	- Creates coordinate frames from peptide bond geometry
	- Direction derived from rotation axis between consecutive frames

	Get the helix direction from 3 consecutive residues.
	Takes 3 dictionaries, containing keys=n,c,ca and value=np_coordinates
	"""

	# Get the peptide bond frames
	frame1 = create_frame_from_three_points(
		res1['C'], res2['N'], res2['CA'])
	frame2 = create_frame_from_three_points(
		res2['C'], res3['N'], res3['CA'])

	return rotation_matrix_to_axis_and_angle(
		np.dot(np.transpose(frame2), frame1))[0]

def normalize(v: np.ndarray) -> np.ndarray:
	"""
	Normalize a vector to unit length using L2 norm.

	Parameters
	----------
	v : np.ndarray
		Input vector

	Returns
	-------
	np.ndarray
		Normalized vector (unit length) or original vector if norm is zero

	Notes
	-----
	Returns original vector unchanged if its norm is zero to avoid division by zero

	Normalize a vector based on its 2 norm.
	"""
	if 0 == np.linalg.norm(v):
		return v
	return v / np.linalg.norm(v)

def create_frame_from_three_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
	"""
	Create a left-handed coordinate frame from three points.

	Constructs an orthonormal coordinate system with p2 as the origin.

	Parameters
	----------
	p1 : np.ndarray
		First point (3D coordinates)
	p2 : np.ndarray
		Second point, used as origin (3D coordinates)
	p3 : np.ndarray
		Third point (3D coordinates)

	Returns
	-------
	np.ndarray
		3x3 matrix where rows are the x, y, z axis unit vectors

	Notes
	-----
	- p2 is the origin
	- y-axis is the normalized vector from p2 to p3
	- z-axis is the normalized cross product of (p1-p2) and y-axis
	- x-axis is the cross product of y and z

	Create a left-handed coordinate frame from 3 points.
	The p2 is the origin; the y-axis is the vector from p2 to p3;
	the z-axis is the cross product of the vector from p2 to p1
	and the y-axis.

	Return a matrix where the axis vectors are the rows.
	"""

	y = normalize(p3 - p2)
	z = normalize(np.cross(p1 - p2, y))
	x = np.cross(y, z)
	return np.array([x, y, z])

def rotation_matrix_to_axis_and_angle(M: np.ndarray) -> tuple[np.ndarray, float]:
	"""
	Calculate rotation axis and angle from a rotation matrix.

	Extracts the axis of rotation and rotation angle from a 3x3 rotation matrix.

	Parameters
	----------
	M : np.ndarray
		3x3 rotation matrix

	Returns
	-------
	tuple[np.ndarray, float]
		A tuple containing:
		- axis : np.ndarray
			Normalized rotation axis vector (3D)
		- angle : float
			Rotation angle in radians

	Notes
	-----
	Uses Rodrigues' rotation formula to extract axis and angle

	Calculate the axis and angle of a rotation matrix.
	"""
	u = np.array([M[2][1] - M[1][2],
		M[0][2] - M[2][0],
		M[1][0] - M[0][1]])

	sin_theta = np.linalg.norm(u) / 2
	cos_theta = (np.trace(M) - 1) / 2

	return normalize(u), np.arctan2(sin_theta, cos_theta)

# Write a function here to calculate the average point in a helix
def centroid_of_residues(resis: list) -> tuple[float, float, float]:
	"""
	Calculate the geometric centroid of residue backbone atoms.

	Computes the average x, y, z coordinates across all backbone atoms
	in the provided residues.

	Parameters
	----------
	resis : list of dict
		List of residue dictionaries, each containing backbone atom keys
		('N', 'CA', 'C', etc.) mapped to 3D coordinate arrays

	Returns
	-------
	tuple of float
		(x, y, z) coordinates of the centroid

	Notes
	-----
	- Includes all atoms present in residue dictionaries
	- Useful for calculating helix displacement to reference points

	Add the x, y, and z coordinates for each residue's backbone atoms
	then return the average value for each dimension (in a tuple).
	"""
	num_atoms = 0
	x = 0
	y = 0
	z = 0
	for res in resis:
		for key in res:
			num_atoms+=1
			x+=res[key][0]
			y+=res[key][1]
			z+=res[key][2]

	return (x/num_atoms, y/num_atoms, z/num_atoms)

### End helix direction and centroid calculating functions.


def get_helical_residues_from_stretch(model, model_pdb_path, start_idx, stop_idx):
	"""
	Calculate the DSSP for the model, then return the start and stop indices
	for the helix contained within the given start_idx and stop_idx
	"""
	# helix_codes = ['H', 'G']
	# helix_codes = ['H']
	# strand_codes = ['E', 'SE']
	# loop_codes = ['T','-', 'B', 'S-']

	helix_codes = ['H', 'G']
	strand_codes = ['E', 'SE', 'B']
	# There are a few cases where several S's in a row (probably) denote a
	# helix, but most of the time it makes sense to count S as a loop
	loop_codes = ['T','-', 'S-', 'S']

	# Not sure why DSSP() has to take a pdb file, oh well.
	# ResIdx starts at 2, but DSSP is 0-indexed
	dssp = DSSP(model, model_pdb_path)
	ss_codes = [dssp[key][2] for key in list(dssp.keys())]

	lhl_ss_codes = []
	# +1 because the stop_idx is included in the bb_rem_res's
	# -1 because DSSP is 0-indexed
	for i in range(start_idx-1, stop_idx+1):
		lhl_ss_codes.append(ss_codes[i])

	helical_residues = []
	helix_start = False
	for idx in range(len(lhl_ss_codes)-1, -1, -1): # starting from the stop_res and working backwards.
		if lhl_ss_codes[idx] in helix_codes:
			# store the residue indices of the original model
			helical_residues.append(start_idx + idx)
			helix_start = True
		elif helix_start == True and len(helical_residues) > 3:
			break

	if len(helical_residues) == 0:
		raise Exception('No helical residues found in residue stretch {0} - {1} in {2} '.format(start_idx, stop_idx, model_pdb_path))

	return helical_residues[-1], helical_residues[0]

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
	"""
	Calculate angle between two vectors in degrees.

	Computes the angle using the dot product of normalized vectors,
	ensuring numerical stability with clipping.

	Parameters
	----------
	v1 : np.ndarray
		First vector (n-dimensional)
	v2 : np.ndarray
		Second vector (n-dimensional, same dimension as v1)

	Returns
	-------
	float
		Angle between vectors in degrees (range: 0-180)

	Notes
	-----
	- Uses arccos with clipping to avoid numerical errors
	- Result is always positive (undirected angle)

	Takes two n-dim vectors and returns the angle between them in degrees
	"""
	v1_u = v1 / np.linalg.norm(v1)
	v2_u = v2 / np.linalg.norm(v2)
	angle_in_radians = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
	angle_in_degrees = angle_in_radians * 180 / np.pi
	return angle_in_degrees

def displacement_between_centroids(c1: tuple, c2: tuple) -> float:
	"""
	Calculate Euclidean distance between two centroids.

	Computes the L2 norm (Euclidean distance) between two points.

	Parameters
	----------
	c1 : tuple of float
		First centroid coordinates (x, y, z)
	c2 : tuple of float
		Second centroid coordinates (x, y, z)

	Returns
	-------
	float
		Euclidean distance between the centroids

	Notes
	-----
	Works with n-dimensional coordinates, not just 3D
	"""
	disp = 0
	for i in range(len(c1)):
		disp += (c1[i] - c2[i])**2
	disp = disp / len(c1)
	disp = disp ** (1/2)
	return disp

def calc_rmsd_on_subset_of_res(ref_struct, moving_struct, atom_types, align_res_subset, \
	rmsd_res_subset):
	"""
	Align two models based on a subset of residues, and calculate the Ca RMSD
	between the rmsd_res_subset residues
	"""
	ref_model = ref_struct[0]
	ref_chain = ref_model["A"]
	moving_model = moving_struct[0]
	moving_chain = moving_model["A"]
	sup = Superimposer()

	# Load lists of Atom objects that will be used to align
	ref_align_atoms, moving_align_atoms = [], []
	for res_idx in align_res_subset:
		for atom_type in atom_types: # ex. 'CA'
			ref_align_atoms.append(ref_chain[res_idx][atom_type])
			moving_align_atoms.append(moving_chain[res_idx][atom_type])

	sup.set_atoms(ref_align_atoms, moving_align_atoms)
	# Apply the superimposition to the moving_atoms
	sup.apply(moving_struct.get_atoms())

	# Load lists of coordinates that will be used to calc RMSD
	# This must be run after the transformation is applied to the moving_struct
	ref_rmsd_coords, moving_rmsd_coords = [], []
	for res_idx in rmsd_res_subset:
		for atom_type in atom_types: # ex. 'CA'
			ref_rmsd_coords.append(ref_chain[res_idx][atom_type].get_coord())
			moving_rmsd_coords.append(moving_chain[res_idx][atom_type].get_coord())

	return calc_RMSD(ref_rmsd_coords, moving_rmsd_coords)

def get_helix_direction_from_residue_stretch(res_idxs, pose):
	"""
	Calculates a helix direction vector given helix
	residue indices and a pyrosetta pose.
	"""
	resis = []
	resis_idxs = []
	for res_idx in res_idxs:
		res = {}
		try:
			for atom in ['N','CA','C']:
				res[atom] = np.array(pose.residue(res_idx).xyz(atom))
			resis.append(res)
			resis_idxs.append(res_idx)
		except:
			print(f"Residue {res_idx} Atom {atom} missing from structure. \
Skipping in helix angle calculation.")

	direction_vector = find_avg_helix_direction(resis, resis_idxs)

	return direction_vector

def get_resis(chain, res_idxs):
	"""
	Return a list of dictionaries. Each dictionary represents
	one residue, with keys=bb atoms and values=atom coords
	"""
	resis = []
	for res_idx in res_idxs:
		res = {}
		try:
			for atom in ['N','CA','C']:
				res[atom] = np.array(chain[res_idx][atom].get_coord())
			resis.append(res)
		except:
			print(f"Residue {res_idx} Atom {atom} missing from structure. \
Skipping in helix displacement calculation.")

	return resis


def get_helix_displacements(
		pose,
		struct_path, # for getting residue coordinates -- this can be replaced with a pyrosetta function, but then PyRosetta residue numbers will need to be used. This should only matter if the start residue index is not 1, or if there are chain breaks.
		helix_residues,
		chain_id='A',
		struct_res_start_idx=1,
		verbose=True,
		sheet_residues=None,
	):
	"""
	Calculate the helix displacements from the centroid of helix residues
	to the centroid of beta sheet residues.
	"""

	# Load a Biopython structure object
	parser = PDBParser()
	parser_struct = parser.get_structure("struct", struct_path)

	# Find beta sheet residues
	# This function uses mkdssp, which needs to be installed.
	# sheet_res_idxs = get_sheet_residues(parser_struct, struct_path)

	if sheet_residues == None:
		# Find beta sheet residues 
		dssp_str = rosetta.core.scoring.dssp.Dssp(pose).get_dssp_secstruct()
		sheet_res_idxs = [i+struct_res_start_idx for i in range(len(dssp_str)) if dssp_str[i]=='E']

		if verbose:
			print(f"DSSP string: {dssp_str}")
	else:
		# sheet_residues have been converted to PyRosetta pose numbering in degree_reshaped.py
		# So, convert them back to PDB numbering with struct_res_start_idx
		# -1 because sheet_residues is already 1-indexed
		sheet_res_idxs = [i+struct_res_start_idx-1 for lst in sheet_residues for i in lst]

	if verbose:
		print(f"Sheet residues: {sheet_res_idxs}")

	model = parser_struct[0]
	chain = model[chain_id]

	# Load coordinates for beta sheet residues
	sheet_resis = get_resis(chain, sheet_res_idxs)

	# Calculate the centroid of the sheet residues
	sheet_centroid = centroid_of_residues(sheet_resis)

	if verbose:
		print(f"Sheet Centroid: {sheet_centroid}")

	# Iterate over lists of helix residues and calculate the displacement 
	# between their centroids and the beta sheet centroid
	helix_displacements = []
	for helix_res_list in helix_residues:
		# Load coordinates for the current helix residues
		helix_resis = get_resis(chain, [i+struct_res_start_idx-1 for i in helix_res_list])

		# Calculate the centroid of the helix residues
		helix_centroid = centroid_of_residues(helix_resis)

		# Calculate the displacement between the helix centroid and sheet centroid
		helix_disp = displacement_between_centroids(helix_centroid, sheet_centroid)

		if verbose:
			print(f"Helix Centroid: {helix_centroid}")
			print(f"Helix Displacement: {helix_disp}")

		helix_displacements.append(helix_disp)

	return helix_displacements

def get_helix_displacements_to_ss(
				start_struct_path,
				design_struct_path, # Used for getting residue coordinates.
				pdb_ss_helix_residues,
				pdb_fd_helix_residues,
				ss_chain_id='A',
				design_chain_id='A',
				ss_res_start_idx=1,
				design_res_start_idx=1,
				verbose=False):

	"""
	Calculate the helix displacements between the centroid of the design structure's
	helix residues to the centroid of the starting structure's helix residues.
	"""

	# Load a Biopython structure object
	parser = PDBParser()
	start_struct = parser.get_structure("start_struct", start_struct_path)
	design_struct = parser.get_structure("design_struct", design_struct_path)

	ss_model = start_struct[0]
	ss_chain = ss_model[ss_chain_id] # for non-chain A structures, call this function with the appropriate chain ID

	design_model = design_struct[0]
	design_chain = design_model[design_chain_id] # for non-chain A structures, call this function with the appropriate chain ID

	# Iterate over lists of helix residues and calculate the displacement 
	# between their centroids and the beta sheet centroid
	helix_displacements_to_ss = []
	for ss_helix_res_list, design_helix_res_list in zip(pdb_ss_helix_residues, pdb_fd_helix_residues):
		# Load coordinates for the current helix residues
		ss_helix_resis = get_resis(ss_chain, [i+ss_res_start_idx-1 for i in ss_helix_res_list])
		design_helix_resis = get_resis(design_chain, [i+design_res_start_idx-1 for i in design_helix_res_list])

		# Calculate the centroid of the helix residues
		ss_helix_centroid = centroid_of_residues(ss_helix_resis)
		design_helix_centroid = centroid_of_residues(design_helix_resis)

		# Calculate the displacement between the helix centroid and sheet centroid
		helix_disp = displacement_between_centroids(ss_helix_centroid, design_helix_centroid)

		if verbose:
			print(f"Helix Centroids: {ss_helix_centroid}, {design_helix_centroid}")
			print(f"Helix Displacement: {helix_disp}")

		helix_displacements_to_ss.append(helix_disp)

	return helix_displacements_to_ss

def get_helix_angles(
		pose,
		helix_residues,
		chain_id='A',
		struct_res_start_idx=1,
		verbose=True,
		sheet_residues=None,
		sheet_dir_vect=[],
		):
	"""
	Calculate the helix angles between helix direction vectors and a 
	reference frame defined by the beta sheet residues.
	"""

	# Find beta sheet residues
	# This function uses mkdssp, which needs to be installed.
	# sheet_res_idxs = get_sheet_residues(parser_struct, struct_path)

	if sheet_residues == None:
		# Find beta sheet residues 
		dssp_str = rosetta.core.scoring.dssp.Dssp(pose).get_dssp_secstruct()
		sheet_res_idxs = [i+struct_res_start_idx for i in range(len(dssp_str)) if dssp_str[i]=='E']

		if verbose:
			print(f"DSSP string: {dssp_str}")
	else:
		sheet_res_idxs = [i+struct_res_start_idx-1 for lst in sheet_residues for i in lst]

	if verbose:
		print(f"Sheet residues: {sheet_res_idxs}")

	if len(sheet_dir_vect) == 0:
		# Calculate the direction vector over the beta sheet residues.
		sheet_dir_vect = get_helix_direction_from_residue_stretch(sheet_res_idxs, pose)

	if verbose:
		print(f"Sheet Direction Vector: {sheet_dir_vect}")

	# Iterate over lists of helix residues and calculate the
	# angle between their direction vectors and the beta sheet
	# direction vector.
	helix_angles = []
	for helix_res_list in helix_residues:
		# Calculate the direction vector over the current helix residues.
		helix_dir_vect = get_helix_direction_from_residue_stretch([i+struct_res_start_idx-1 for i in helix_res_list], pose)

		helix_angle = angle_between(helix_dir_vect, sheet_dir_vect)

		if verbose:
			print(f"Helix Direction Vector: {helix_dir_vect}")
			print(f"Helix Angle: {helix_angle}")

		helix_angles.append(helix_angle)

	return helix_angles

def get_helix_angles_to_ss(
		ss_pose,
		design_pose,
		ss_helix_residues,
		design_helix_residues,
		ss_chain_id='A',
		design_chain_id='A',
		ss_struct_res_start_idx=1,
		design_struct_res_start_idx=1,
		verbose=True
		):
	"""
	Calculate the angles between helices in the aligned 
	starting and design structures.
	"""

	# Iterate over lists of helix residues and calculate the displacement 
	# between their centroids and the beta sheet centroid
	helix_angles_to_ss = []
	for ss_helix_res_list, design_helix_res_list in zip(ss_helix_residues, design_helix_residues):
		# Calculate the direction vector over the current helix residues.
		ss_helix_dir_vect = get_helix_direction_from_residue_stretch([i+ss_struct_res_start_idx-1 for i in ss_helix_res_list], ss_pose)
		design_helix_dir_vect = get_helix_direction_from_residue_stretch([i+design_struct_res_start_idx-1 for i in design_helix_res_list], design_pose)

		helix_angle = angle_between(ss_helix_dir_vect, design_helix_dir_vect)

		if verbose:
			print(f"Helix Direction Vectors: {ss_helix_dir_vect}, {design_helix_dir_vect}")
			print(f"Helix Angle: {helix_angle}")

		helix_angles_to_ss.append(helix_angle)

	return helix_angles_to_ss

def update_df_with_LUCS_metrics(df):
	"""
	Add the LUCS metrics to the given df.
	Return the updated df.

	Previously, I loaded the columns based on whatever was in the filter_info.json
	dicts. I started using a list of expected columns instead for ease of dealing with
	designs that are missing filter_info.json files.
	"""
	columns = ['designable_residues_average_energy',
				'movable_residues_average_energy',
				'all_average_energy',
				'designable_residues_max_energy',
				'movable_residues_max_energy',
				'all_residues_max_energy',
				'num_over_saturated_hbond_acceptors_for_designable_residues',
				'num_over_saturated_hbond_acceptors_for_movable_residues',
				'num_over_saturated_hbond_acceptors_for_all_residues',
				'hydrophobic_sasa_sc_for_designable_residues',
				'hydrophobic_sasa_for_movable_residues',
				'average_hydrophobic_sasa_for_movable_residues',
				'relative_hydrophobic_sasa_for_movable_residues',
				'buried_nonpolar_surface_area_per_res',
				'designable_local_holes_score',
				'movable_local_holes_score',
				'all_local_holes_score',
				'bb_buried_unsat_for_bb_remodeled_residues',
				'buried_unsat_for_designable_residues',
				'buried_unsat_for_movable_residues',
				'buried_unsat_for_all_residues',
				'bb_backrub_ensemble_buhs_for_bb_remodeled_residues',
				'backrub_ensemble_consensus_buhs_for_designable_residues',
				'backrub_ensemble_consensus_buhs_for_movable_residues',
				'backrub_ensemble_consensus_buhs_for_all_residues',
				'bb_remodeled_residues_helix_complementarity',
				'bb_remodeled_worst_fragment_crmsd',
				'bb_remodeled_mean_fragment_crmsd',
				'linker_ids',
				'model_id']

	new_df_columns = {}
	for col in columns:
		new_df_columns[col] = []

	skip_cols = []
	set_none = False
	for index, row in df.iterrows():
		design_id = row['design_id']
		filter_info_path = os.path.join(row['lucs_location'], 'filter_info.json')
		try:
			with open(os.path.join(row['lucs_location'], 'filter_info.json')) as filter_info_json:
				filter_info = json.load(filter_info_json)
		except:
			print(f'Warning: {filter_info_path} not found. Adding None value Rosetta metrics for {design_id}.')
			set_none = True

		for col in columns:
			try:
				if set_none: new_df_columns[col].append(None)
				elif col == 'model_id':
					new_df_columns[col].append(int(filter_info[col]))
				else:
					try:
						# If value is float
						new_df_columns[col].append(float(filter_info[col]))
					except:
						# If value is string or other type of Series
						new_df_columns[col].append(filter_info[col])
			except:
				print(f'Warning: {col} not found in filter_info.json. Skipping this column.')
				skip_cols.append(col)
				continue

		set_none = False

	for col in new_df_columns:
		if col not in skip_cols:
			df[col] = new_df_columns[col]
			print(col)


	return df

def update_df_with_best_af_model(df, df_col_prefix_values, df_col_prefix_models, \
	af_metric, af_metric_low_better, add_best_af_metrics=False):
	"""
	Update the df with columns denoting the best AF2/3 model and
	best AF2/3 model's metric that was used to decide which model was best.

	af_metric is the prefix to an af_metric. For example, for
	rmsd_reshaped_CA_fd_to_af_model1:
	af_metric = 'rmsd_reshaped_CA_fd_to_af_'
	df_col_prefix_values = 'best_af_value_'
	df_col_prefix_models = 'best_af_model_'
	"""
	best_models = []
	best_values = []
	af_models = ['model1','model2','model3','model4','model5']
	for index, row in df.iterrows():
		if af_metric_low_better:
			best_af_value = 999
		else:
			best_af_value = -1
		best_af_model = None
		for model in af_models:
			if af_metric_low_better:
				if row[af_metric+model] < best_af_value:
					best_af_value = row[af_metric+model]
					best_af_model = model
			else:
				if row[af_metric+model] > best_af_value:
					best_af_value = row[af_metric+model]
					best_af_model = model
		best_values.append(best_af_value)
		best_models.append(best_af_model)
	# af_metric[:-1] gets rid of the trailing _
	df[df_col_prefix_values+af_metric[:-1]] = best_values
	df[df_col_prefix_models+af_metric[:-1]] = best_models
	if add_best_af_metrics:
		df = add_best_af_metrics_to_df(df, df_col_prefix_values, \
			df_col_prefix_models, af_metric)
	return df

def add_best_af_metrics_to_df(df, df_col_prefix_values, \
	df_col_prefix_models, af_metric):
	"""
	Add columns to the df with LUCS-AF metrics for the 'best AF model'
	w.r.t. the provided af_metric. Return the updated df.

	Metrics to add for best AF model: 1. CaRMSD all residues, 2. CaRMSD reshaped
	residues, 3. pLDDT all residues, 4. pLDDT reshaped residues, 5. CaRMSD LUCS model to
	starting structure (can't do this without selecting subset of residues),
	6. CaRMSD best AF model to starting structure (again, can't do this
	without selecting subset to compare), 7. TM score LUCS model to starting structure,
	and 8. TM score best AF model to starting structure.

	df_col_prefix_values = 'best_af_value_'
	df_col_prefix_models = 'best_af_model_'
	af_metric = 'all_plddt_af_'
	"""
	# The best_af_model_metrics that end in '_' are df column names
	# that end in 'model[1,2,3,4,5]'
	best_af_model_metrics = ['rmsd_all_res_CA_fd_to_af_',
		'rmsd_reshaped_CA_fd_to_af_','ave_plddt_af_','ave_bb_rem_plddt_af_',
		'ave_non_bb_rem_plddt_af_','TM_score_fd_to_ss','TM_score_ss_to_af_']

	# ex/ new_col_header=best_af_model_by_all_plddt_af_ is the best
	# AF model as judged by ave all residue pLDDT.
	new_col_header = df_col_prefix_models+'by_'+af_metric

	# Initialize dict that will contain new col values
	new_col_dict = {}
	for metric in best_af_model_metrics:
		new_col_dict[new_col_header+'value_for_'+metric] = []

	# Fill the new columns with LUCS-AF values for the
	# best AF model (best w.r.t. af_metric)
	for index, row in df.iterrows():
		# af_metric[:-1] gets rid of the trailing _
		best_af_model = row[df_col_prefix_models+af_metric[:-1]]

		# Get the AF-LUCS metrics for the best model
		for metric in best_af_model_metrics:

			# If AF models were not found for the current design,
			# leave these best_af_model columns empty
			if best_af_model == None:
				full_metric = None
			elif metric[-1]=='_':
				full_metric = metric+best_af_model
			else: # for when metric=='TM_score_fd_to_ss'
				full_metric = metric

			# If AF models were not found for the current design,
			# leave these best_af_model columns empty
			if full_metric == None:
				new_col_dict[new_col_header+'value_for_'+metric].append(None)
			else:
				new_col_dict[new_col_header+'value_for_'+metric].append(row[full_metric])

	for new_col in new_col_dict:
		df[new_col] = new_col_dict[new_col]

	return df

def calc_TM_score(pdb_path_1: str, pdb_path_2: str) -> float:
	"""
	Calculate TM-score between two pdb structures.

	Uses tmtools to compute the Template Modeling score, a metric for
	assessing structural similarity between two protein structures.

	Parameters
	----------
	pdb_path_1 : str
		Path to first pdb file
	pdb_path_2 : str
		Path to second pdb file

	Returns
	-------
	float
		TM-score normalized by the length of chain 1 (range: 0-1)

	Notes
	-----
	- TM-score > 0.5 generally indicates same fold
	- TM-score > 0.17 is better than random
	- Uses the first chain from each structure

	See Also
	--------
	get_TM_score : Calculate TM-score from pre-loaded chain objects

	Use tmtools to calculate the TM score between two PDB structures.
	"""
	tm_s1 = get_structure(pdb_path_1)
	tm_s2 = get_structure(pdb_path_2)
	tm_s1_chain = next(tm_s1.get_chains())
	tm_s2_chain = next(tm_s2.get_chains())
	return get_TM_score(tm_s1_chain, tm_s2_chain)

def get_TM_score(chain_1, chain_2) -> float:
	"""
	Calculate TM-score between two pre-loaded tmtools chain objects.

	Computes Template Modeling score from tmtools chain objects that have
	already been loaded from PDB structures.

	Parameters
	----------
	chain_1 : tmtools chain object
		First protein chain
	chain_2 : tmtools chain object
		Second protein chain

	Returns
	-------
	float
		TM-score normalized by the length of chain 1 (range: 0-1)

	Notes
	-----
	- Extracts coordinates and sequences from chain objects
	- Performs TM-align algorithm to compute score
	- TM-score is length-independent and normalized to [0,1]

	See Also
	--------
	calc_TM_score : Calculate TM-score directly from pdb file paths

	Use tmtools to calculate and return the TM score between two tmtools chains.
	"""
	coords_1, seq_1 = get_residue_data(chain_1)
	coords_2, seq_2 = get_residue_data(chain_2)
	res = tm_align(coords_1, coords_2, seq_1, seq_2)

	return res.tm_norm_chain1

def truncate_helical_residues(ss_helix1_residxs, ss_helix2_residxs, fd_helix1_residxs, fd_helix2_residxs):
	"""
	Truncate the longer helix to the length of the shorter helix.
	If one length is odd and the other is even, keep the N-terminal residue of the
	longer helix.
	"""
	truncated_reslists = []
	for res_list1, res_list2 in [(list(range(ss_helix1_residxs[0],ss_helix1_residxs[1]+1)), list(range(fd_helix1_residxs[0],fd_helix1_residxs[1]+1))), \
		(list(range(ss_helix2_residxs[0],ss_helix2_residxs[1]+1)), list(range(fd_helix2_residxs[0],fd_helix2_residxs[1]+1)))]:
		i=0
		if len(res_list1) == len(res_list2):
			pass
		elif len(res_list1) > len(res_list2):
			while(len(res_list1) > len(res_list2)):
				if i // 2 == 0:
					res_list1 = res_list1[:-1] # remove the C-terminal residue first.
				else:
					res_list1 = res_list1[1:]
				i+=1
		else:
			while(len(res_list2) > len(res_list1)):
				if i // 2 == 0:
					res_list2 = res_list2[:-1] # remove the C-terminal residue first.
				else:
					res_list2 = res_list2[1:]
				i+=1
		truncated_reslists.append(res_list1)
		truncated_reslists.append(res_list2)
	
	return [sublist for sublist in truncated_reslists]

def truncate_sheet_residues(fd_sheet_residxs, ss_sheet_residxs):
	"""
	Truncate the longer list of sheet residues to the length of the shorter list.
	If one length is odd and the other is even, keep the N-terminal residue of the
	longer list.
	"""
	truncated_reslists = []
	for res_list1, res_list2 in [(fd_sheet_residxs, ss_sheet_residxs)]:
		i=0
		if len(res_list1) > len(res_list2):
			while(len(res_list1) > len(res_list2)):
				if i // 2 == 0:
					res_list1 = res_list1[:-1] # remove the C-terminal residue first.
				else:
					res_list1 = res_list1[1:]
				i+=1
		else:
			while(len(res_list2) > len(res_list1)):
				if i // 2 == 0:
					res_list2 = res_list2[:-1] # remove the C-terminal residue first.
				else:
					res_list2 = res_list2[1:]
				i+=1
		truncated_reslists.append(res_list1)
		truncated_reslists.append(res_list2)
	
	return [sublist for sublist in truncated_reslists]

def get_strand_dir_vec(pose,
		sheet_residues,
		chain_id='A'):
	"""
	Takes a PyRosetta pose object (struct)

	Return a direction vector for the beta sheet residues, using the 
	central residues from each beta strand.
	"""

	all_strand_dir_vects = []
	for strand_residues in sheet_residues:
		try:
			# Calculate the direction vector over the beta strand residues in the starting structure.
			ss_central_strand_res = strand_residues[len(strand_residues)//2]
			# Simply subtract the strand's N-terminal CA atom from its C-terminal CA atom
			nterm_coord = np.array(pose.residue(ss_central_strand_res-1).xyz("CA"))
			cterm_coord = np.array(pose.residue(ss_central_strand_res+1).xyz("CA"))
			strand_dir_vect = cterm_coord - nterm_coord
			all_strand_dir_vects.append(strand_dir_vect)
		except:
			print(f"Warning: Beta Strand direction vector could not be loaded for strand residues: {strand_residues}.\nSkipping this beta strand in helix angle calculation.")

	# Ensure that each strand has the same N->C direction
	angles_between = np.zeros((len(all_strand_dir_vects), len(all_strand_dir_vects)))
	for i in range(len(all_strand_dir_vects)-1):
		for j in range(i+1, len(all_strand_dir_vects)):
			angles_between[i][j] = angle_between(all_strand_dir_vects[i], all_strand_dir_vects[j])

	# Exclude strands with opposite directions to neighboring 
	# arbitrarily set cutoff for opposite directions at > 100 deg
	exclude_idxs = []
	for i in range(len(all_strand_dir_vects)-1):
		if i == 0:
			if angles_between[i][i+1] > 100 and not angles_between[i+1][i+2] > 100:
				exclude_idxs.append(i)
		elif i == len(all_strand_dir_vects)-1:
			if angles_between[i][i+1] > 100 and not angles_between[i-1][i] > 100:
				exclude_idxs.append(i+1)
		else:
			if angles_between[i-1][i] > 100 and angles_between[i][i+1] > 100:
				exclude_idxs.append(i)

	final_strand_dir_vects = []
	for i in range(len(all_strand_dir_vects)):
		if i not in exclude_idxs:
			final_strand_dir_vects.append(all_strand_dir_vects[i])

	return np.mean(np.array(final_strand_dir_vects), axis=0)

def get_sheet_residues(struct, 
		pdb_start_idx=1,
		strand_nums=None,
		all_sheet_residues=None):
	"""
	Return a list of lists of beta strand residue indices.

	struct is a PyRosetta pose object
	"""
	# Load lists of the Starting Structure and Design Model sheet residues, which are
	# used in calculating a direction vector for the sheet residues, used later in helix
	# angle calculations.
	if all_sheet_residues == None:
		# Load a list of sheet_residues using DSSP
		dssp_str = rosetta.core.scoring.dssp.Dssp(struct).get_dssp_secstruct()
		dssp_all_sheet_residues = [i+pdb_start_idx-1 for i in range(len(dssp_str)) if dssp_str[i]=='E']
		# Get just the second beta strand
		split_sheet_res = [[]]
		for i in range(len(dssp_all_sheet_residues)-1):
			if dssp_all_sheet_residues[i]+1 == dssp_all_sheet_residues[i+1]:
				split_sheet_res[-1].append(dssp_all_sheet_residues[i])
			else:
				split_sheet_res.append([])
		sheet_residues = []
		if strand_nums == None:
			sheet_residues = split_sheet_res
		else:
			for strand_idx in strand_nums:
				sheet_residues.append(split_sheet_res[strand_idx])

	else:
		sheet_residues = []
		if strand_nums == None:
			sheet_residues = all_sheet_residues
		else:
			for strand_idx in strand_nums:
				sheet_residues.append(all_sheet_residues[strand_idx])

	return sheet_residues


#####
### Helix RMSD functions
#####

def load_design(args, design_path, bb_remodeled_residues=None):
	'''Load a design.
	Return:
		pose_design, pose_to_compare, pose_to_compare_type, bb_remodeled_residues, bb_fixed_residues
	'''
	pose_design = rosetta.core.import_pose.pose_from_file(os.path.join(design_path))

	if bb_remodeled_residues == None:
		with open(os.path.join(design_path.rsplit('/',1)[0], 'design_info.json'), 'r') as f:
			pdb_bb_remodeled_residues_all = sorted(json.load(f)['bb_remodeled_residues'])

		# Convert residue numbers from PDB to PyRosetta numbering
		bb_remodeled_residues_all = convert_pdb_to_pose_resnums(pose_design, pdb_bb_remodeled_residues_all, verbose=args.verbose)

	else:
		## If bb_remodeled_residues is provided, then they are already converted to PyRosetta Pose numbering
		bb_remodeled_residues_all = bb_remodeled_residues

	# Convert all pose residues to pyrosetta numbering
	all_pose_residues = convert_pdb_to_pose_resnums(pose_design, [i+1 for i in range(pose_design.total_residue())], verbose=args.verbose)

	# Separate remodeled residues into segments 

	separators = [i + 1 for i in range(len(bb_remodeled_residues_all) - 1) if bb_remodeled_residues_all[i] + 1 != bb_remodeled_residues_all[i + 1]] 
	separators = [0] + separators + [len(bb_remodeled_residues_all)]

	bb_remodeled_residues = []
	for i in range(len(separators) - 1):
		bb_remodeled_residues.append(list(range(bb_remodeled_residues_all[separators[i]], 
			bb_remodeled_residues_all[separators[i + 1] - 1] + 1)))

	# bb_fixed_residues = [i for i in range(1, pose_design.size() + 1) if not (i in bb_remodeled_residues_all)]
	## After converting to PyRosetta pose numbering
	bb_fixed_residues = [i for i in all_pose_residues if not (i in bb_remodeled_residues_all)]

	if args.verbose:
		print(f'all_pose_residues in load_design:')
		print(all_pose_residues)
		print(f'bb_remodeled_residues_all in load_design:')
		print(bb_remodeled_residues_all)

	return pose_design, pose_design, bb_remodeled_residues, bb_fixed_residues

def xyzV_to_np_array(xyz):
	return np.array([xyz.x, xyz.y, xyz.z])

def np_array_to_xyzV(a):
	return rosetta.numeric.xyzVector_double_t(a[0], a[1], a[2])

def np_array_to_xyzM(a):
	return rosetta.numeric.xyzMatrix_double_t.rows(
			a[0][0], a[0][1], a[0][2],
			a[1][0], a[1][1], a[1][2],
			a[2][0], a[2][1], a[2][2])

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

def get_backbone_points(pose, residues, atom_types=['N', 'CA', 'C']):
	'''Get backbone points for residues in a pose.'''
	points = []

	for res in residues:
		for atom in atom_types:
			try:
				points.append(xyzV_to_np_array(pose.residue(res).xyz(atom)))
			except:
				print(f'Residue {res} not found in pose. Skipping alignment on this residue')
				points.append((-9999,-9999,-9999))

	return points

def superimpose_poses_by_residues(pose_source, residues_source, pose_target, residues_target, atom_types=['N', 'CA', 'C']):
	'''Superimpose residues in a source pose into residues in a target pose.
	Only backbone atoms are used for the superimposition.
	'''

	## If these two residue lists are of different
	## lengths, trim the N-terminal residues of the longer sequence.
	if len(residues_source) != len(residues_target):
		min_len = min(len(residues_source), len(residues_target))
		residues_source = residues_source[-1*min_len:]
		residues_target = residues_target[-1*min_len:]

	# Get the points to be superimposed

	points_source_unfiltered = get_backbone_points(pose_source, residues_source, atom_types=atom_types)
	points_target_unfiltered = get_backbone_points(pose_target, residues_target, atom_types=atom_types)

	points_source, points_target = [], []
	for ps, pt in zip(points_source_unfiltered, points_target_unfiltered):
		if ps[0] == -9999 or pt[0] == -9999:
			continue
		else:
			points_source.append(ps)
			points_target.append(pt)

	points_source = np.array(points_source)
	points_target = np.array(points_target)

	# Get the rigid body transformation

	M, t = get_superimpose_transformation(points_source, points_target)

	# Transform the source pose

	pose_source.apply_transform_Rx_plus_v(np_array_to_xyzM(M), 
			np_array_to_xyzV(t))

	return pose_source, pose_target

def calc_backbone_RMSD(pose1, residues1, pose2, residues2, atom_types=['N', 'CA', 'C']):
	'''Calculate backbone RMSD between two poses for specific positions.'''
	assert(len(residues1) == len(residues2))

	def RMSD(points1, points2):
		'''Calcualte RMSD between two lists of numpy points.'''
		diff = [points1[i] - points2[i] for i in range(len(points1))]
		try:
			return np.sqrt(sum(np.dot(d, d) for d in diff) / len(diff))
		except:
			print('\n\nCurrent design is missing helix residues in its design model! Setting helix RMSD = None for this LHL.\n\n')
			return None

	points1 = get_backbone_points(pose1, residues1, atom_types=atom_types)
	points2 = get_backbone_points(pose2, residues2, atom_types=atom_types)

	return RMSD(points1, points2)

def rmsd_between_segments(pose1, segment1, pose2, segment2):
	'''Calculate backbone RMSD between two segments.
	A segment is defined as (start, stop).
	'''
	# Swap the poses if the length of the second sse is longer than the first one

	if segment2[1] - segment2[0] < segment1[1] - segment1[0]:
		pose1, segment1, pose2, segment2 = pose2, segment2, pose1, segment1

	length1 = segment1[1] - segment1[0] + 1
	length_diff = (segment2[1] - segment2[0]) - (segment1[1] - segment1[0])

	# Calculate RMSDs for different windows

	rmsds = []
	residues1 = list(range(segment1[0], segment1[1] + 1))

	for shift in range(length_diff + 1):
		residues2 = list(range(segment2[0] + shift, segment2[0] + length1 + shift))

		rmsds.append(calc_backbone_RMSD(pose1, residues1, pose2, residues2))

	# Return the lowest RMSD

	return min(rmsds)

def get_helix_of_lhl_unit(pose, lhl_start, lhl_stop):
	"""
	Find the longest helix in an LHL unit.
	Return a pair of (start, stop).

	lhl_start and lhl_stop are 1-indexed, which dssp_str is 0-indexed.
	"""
	dssp_str = rosetta.core.scoring.dssp.Dssp(pose).get_dssp_secstruct()

	max_length = 0
	current_length = 0
	max_start_index = -1

	# lhl_start and lhl_stop are 1-indexed, which dssp_str is 0-indexed.
	for i in range(lhl_start-1, lhl_stop):
		if dssp_str[i] == 'H':
			current_length += 1
			if current_length > max_length:
				max_length = current_length
				max_start_index = i - current_length + 1
		else:
			current_length = 0

	if max_start_index != -1:
		return (max_start_index, max_start_index + max_length - 1)
	else:
		return (-1, -1)

def calculate_bb_remodeled_region_rmsd(args,
		reference_path,
		design_path,
		helix_idxs=[0,1],
		is_af_design=False,
		atom_types=['N', 'CA', 'C'],
		ref_lhl_residues=None,
		pose_lhl_residues=None,
		ref_helix_residues=None,
		pose_helix_residues=None,
		align_residues=None):
	'''
	Calculate the Helix RMSDs between reference and design structures.
	The "reference" will be either a LUCS starting structure or LUCS design, and the "pose" will
	be either a LUCS design or an AF2/3 model.

	bb_remodeled_residues correspond to the design_path design.

	Helix Residues is a list of lists, the first list having
		the length of the number of LHLs, and the nested list
		having the helical residues for each LHL.

	Return the helix RMSD and two lists of lists of the length-matched helix residues used
	''' 

	# Load the designs
	if ref_lhl_residues != None:
		ref_bb_remodeled_residues = [i for lhl in ref_lhl_residues for i in lhl]
		ref_design, ref_pose_to_compare, ref_bb_remodeled_residues, ref_bb_fixed_residues \
			= load_design(args, reference_path, bb_remodeled_residues=ref_bb_remodeled_residues)
	else:
		ref_design, ref_pose_to_compare, ref_bb_remodeled_residues, ref_bb_fixed_residues \
			= load_design(args, reference_path, bb_remodeled_residues=None)

	if is_af_design:
		# Provide the bb_remodeled_residues for the AF2/3 models.
		pose_bb_remodeled_residues = [i for lhl in pose_lhl_residues for i in lhl]
		pose_design, poses_to_compare, all_bb_remodeled_residues, all_bb_fixed_residues \
			= load_design(args, design_path, bb_remodeled_residues=pose_bb_remodeled_residues)
	elif pose_lhl_residues != None:
		pose_bb_remodeled_residues = [i for lhl in pose_lhl_residues for i in lhl]
		pose_design, poses_to_compare, all_bb_remodeled_residues, all_bb_fixed_residues \
			= load_design(args, design_path, bb_remodeled_residues=pose_bb_remodeled_residues)
	else:
		pose_design, poses_to_compare, all_bb_remodeled_residues, all_bb_fixed_residues \
			= load_design(args, design_path, bb_remodeled_residues=None)

	# Superimpose design i to reference design
	if align_residues != None:
		# superimpose_poses_by_residues expects a single list of residues on which to align.
		pose_design, ref_pose_to_compare = superimpose_poses_by_residues(pose_design, [i for lst in align_residues[1] for i in lst], 
			ref_pose_to_compare, [i for lst in align_residues[0] for i in lst], atom_types=atom_types)

	else:
		pose_design, ref_pose_to_compare = superimpose_poses_by_residues(pose_design, all_bb_fixed_residues, ref_pose_to_compare, ref_bb_fixed_residues, atom_types=atom_types)

	if args.aligned_pdb_outdir != None:
		os.makedirs(args.aligned_pdb_outdir, exist_ok=True)
		aligned_pdb_outpath = os.path.join(args.aligned_pdb_outdir, design_path.split('/')[-1].replace('.pdb', '_aligned_to_ss.pdb'))
		pose_design.dump_pdb(aligned_pdb_outpath)
		if args.verbose:
			print(f"Saved aligned PDB to Starting Structure: {aligned_pdb_outpath}")

	# Find the mapped residues between the two structures

	mapped_residues_i = []
	mapped_residues_j = []
	split_mapped_residues_i = []
	split_mapped_residues_j = []

	## Loop over all redesigned LHLs (2 LHLs for the RO2 designs)
	for k in helix_idxs:
		seg_i = all_bb_remodeled_residues[k]
		# seg_j = all_bb_remodeled_residues[j][k]
		seg_j = ref_bb_remodeled_residues[k]

		# Get the helix part of the LHL unit

		# If the pose helix residues are provided (as a list of lists)
		if pose_helix_residues != None:
			helix_i = (pose_helix_residues[k][0], pose_helix_residues[k][-1]) # tuple of start and stop indices.
		else:
			helix_i = get_helix_of_lhl_unit(pose_design, min(seg_i), max(seg_i))
		
		# If the reference helix residues are provided (as a list of lists)
		if ref_helix_residues != None:
			helix_j = (ref_helix_residues[k][0], ref_helix_residues[k][-1]) # tuple of start and stop indices.
		else:
			helix_j = get_helix_of_lhl_unit(ref_pose_to_compare, min(seg_j), max(seg_j))

		len_comp = min(helix_i[1] - helix_i[0], helix_j[1] - helix_j[0])

		h_mid_start1 = (sum(helix_i) - len_comp) // 2
		h_mid_start2 = (sum(helix_j) - len_comp) // 2

		mapped_residues_i += [h_mid_start1 + l for l in range(len_comp)]
		mapped_residues_j += [h_mid_start2 + l for l in range(len_comp)]

		split_mapped_residues_i.append([h_mid_start1 + l for l in range(len_comp)])
		split_mapped_residues_j.append([h_mid_start2 + l for l in range(len_comp)])

	if args.verbose:
		print(f'Helix RMSD will be calculated over the following residues:')
		print(f'Design model residues: {mapped_residues_i}')
		print(f'Starting Structure residues: {mapped_residues_j}')

	# Calculate RMSD over all helix residues
	rmsd = calc_backbone_RMSD(ref_pose_to_compare, mapped_residues_j, pose_design, mapped_residues_i)

	if args.verbose:
		print(f'Helix RMSD: {rmsd}')

	# Returning the split mapped residues for easy storing of 'trimmed_fd_helix{helix_idx}_residues'
	return rmsd, split_mapped_residues_j, split_mapped_residues_i

#####
### End Helix RMSD functions
#####

#####
### PyRosetta RMSD functions
#####

def load_pyrosetta_strings(pdb_path, lhl_residues_list):

	# Load strings containing residue ranges for PyRosetta
	non_bb_remodeled_residues_string_list = []
	bb_remodeled_residues_string_list = []

	non_bb_rem_ranges = []

	pdb_reslist = get_residue_list_from_pdb(pdb_path)

	pose = pose_from_file(pdb_path)

	reslist = convert_pdb_to_pose_resnums(pose, pdb_reslist, verbose=False)

	non_bb_rem_reslist = [i for i in reslist if i not in [j for lst in lhl_residues_list for j in lst]]

	non_bb_rem_separators = [i for i in range(len(non_bb_rem_reslist)-1) if non_bb_rem_reslist[i] + 1 != non_bb_rem_reslist[i+1]]
	
	non_bb_rem_ranges.append((non_bb_rem_reslist[0], non_bb_rem_reslist[non_bb_rem_separators[0]]))
	for i in range(len(non_bb_rem_separators) - 1):
		non_bb_rem_ranges.append((non_bb_rem_reslist[non_bb_rem_separators[i]+1], non_bb_rem_reslist[non_bb_rem_separators[i+1]]))
	non_bb_rem_ranges.append((non_bb_rem_reslist[non_bb_rem_separators[-1]+1], non_bb_rem_reslist[-1]))

	for i, lhl_residues in enumerate(lhl_residues_list):
		bb_remodeled_residues_string_list.append(f"{lhl_residues[0]}-{lhl_residues[-1]}")

	for res_range in non_bb_rem_ranges:
		non_bb_remodeled_residues_string_list.append(f"{res_range[0]}-{res_range[1]}")

	return non_bb_remodeled_residues_string_list, bb_remodeled_residues_string_list

def align_by_residues(args,
		ref_pose,
		mobile_pose,
		ref_res_sel_string_list,
		mobile_res_sel_string_list):
	"""
	Aligns two poses by selected residues.

	res_sel_string is a string passed to the ResidueIndexSelector, such as "1-100",
	which is used to align the two poses.
	"""

	# If one pose is longer than another, then trim the
	# last set of continuous residues by which to align
	# the structures.
	# If align_residues.json files are provided, then get_design_to_ss_aligned_residues()
	# ensures that the same number of residues from each structure is used in alignment.
	if not args.use_align_residues_file:
		if len(ref_pose.sequence()) > len(mobile_pose.sequence()):
			ref_res_sel_string_list[-1] = ref_res_sel_string_list[-1].replace(f'-{len(ref_pose.sequence())}', f'-{len(mobile_pose.sequence())}')
			if args.verbose:
				print(f'Reference pose sequence length does not match mobile pose: {ref_pose.sequence()}\n{mobile_pose.sequence()}')
				print(f'Reference pose sequence length: {len(ref_pose.sequence())}')
				print(f'Mobile pose sequence length: {len(mobile_pose.sequence())}')
		elif len(ref_pose.sequence()) < len(mobile_pose.sequence()):
			mobile_res_sel_string_list[-1] = mobile_res_sel_string_list[-1].replace(f'-{len(mobile_pose.sequence())}', f'-{len(ref_pose.sequence())}')
			if args.verbose:
				print(f'Reference pose sequence length does not match mobile pose: {ref_pose.sequence()}\n{mobile_pose.sequence()}')
				print(f'Reference pose sequence length: {len(ref_pose.sequence())}')
				print(f'Mobile pose sequence length: {len(mobile_pose.sequence())}')

	# Select residues in ref pose
	ref_res = OrResidueSelector()

	# Iterate over all continuous stretches of residue indices
	for res_sel_string in ref_res_sel_string_list:
		ref_res_idxs = ResidueIndexSelector(res_sel_string)
		ref_res.add_residue_selector(ref_res_idxs)

	# Select residues in moving pose
	mobile_res = OrResidueSelector()

	# Iterate over all continuous stretches of residue indices
	for res_sel_string in mobile_res_sel_string_list:
		mobile_res_idxs = ResidueIndexSelector(res_sel_string)
		mobile_res.add_residue_selector(mobile_res_idxs)

	if args.verbose:
		print(f"Aligning mobile pose residues: {mobile_res}")
		print(f"to reference pose residues: {ref_res}")

	abr = AlignByResidueSelectorMover()
	abr.query_selector(mobile_res)
	abr.reference_pose(ref_pose)
	abr.reference_selector(ref_res)
	abr.apply(mobile_pose)

	return ref_pose, mobile_pose

def calculate_RMSD(ref_pose,
		mobile_pose,
		rmsd_res_string_list,
		rmsd_type=pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_heavy_including_O):
	"""
	Takes two already-aligned poses and returns bb RMSDs.

	rmsd_res_string_list is a list of strings passed to the ResidueIndexSelector,
	such as "1-100". An RMSD will be calculated over each set of residues.

	Currently, the script returns the average of the RMSDs, but individual RMSDs could also be returned.
	"""

	# If one pose is longer than another, then trim the
	# last set of continuous residues by which to align
	# the structures.
	if len(ref_pose.sequence()) > len(mobile_pose.sequence()) and \
		str(len(ref_pose.sequence())) in rmsd_res_string_list[-1]:
		rmsd_res_string_list[-1] = \
			rmsd_res_string_list[-1].replace(f'-{len(ref_pose.sequence())}', f'-{len(mobile_pose.sequence())}')

	rmsd_list = []

	for rmsd_res_string in rmsd_res_string_list:
		rmsd_res = ResidueIndexSelector(rmsd_res_string)

		# Set up the RMSDMetric
		rmsd_metric = RMSDMetric()
		rmsd_metric.set_run_superimpose(False)
		rmsd_metric.set_comparison_pose(ref_pose)
		rmsd_metric.set_rmsd_type(rmsd_type)

		# Calculate RMSD over the rmsd_res residues.
		rmsd_metric.set_residue_selector(rmsd_res)
		rmsd_metric.set_residue_selector_reference(rmsd_res)
		rmsd = rmsd_metric.calculate(mobile_pose)

		# Could alternatively use a residue mapping instead of two selectors, but I think the above is clearer
		# sel_map = selectors.get_residue_mapping_from_selectors(pose_latch_sel, ref_latch_sel, pose, ref_pose)
		# new_rmsd_metric.set_residue_mapping(sel_map)

		rmsd_list.append(rmsd)

	return np.mean(rmsd_list)

#####
### End PyRosetta RMSD functions
#####


def get_remodeled_residues_from_insertion_points(
		insertion_points_file, 
		res_start_idx,
		seq_len):

	with open(insertion_points_file) as f:
		insertion_points = json.load(f)

	start_stop_idxs = []
	for d in insertion_points:
		start_stop_idxs.append((d['start'],d['stop']))

	bb_remodeled_residues, non_bb_remodeled_residues = [], []

	for start, stop in start_stop_idxs:
		for i in range(start, stop+1):
			bb_remodeled_residues.append(i)

	# seq_len+res_start_idx to include the last residue,
	# and generate lists with residue indices that match the
	# input structure.
	for i in range(res_start_idx, seq_len+res_start_idx):
		if i not in bb_remodeled_residues:
			non_bb_remodeled_residues.append(i)

	return bb_remodeled_residues, non_bb_remodeled_residues, start_stop_idxs

def get_remodeled_residues_from_design_info(
		pdb_file,
		design_info_file):

	# Load the list of bb_remodeled_residues
	with open(design_info_file) as design_info_json:
		design_info = json.load(design_info_json)
		bb_remodeled_residues = design_info['bb_remodeled_residues']

	reslist = get_residue_list_from_pdb(pdb_file)

	# Load a list of non-bb_remodeled residue indices.
	non_bb_remodeled_residues = []
	for res_idx in reslist:
		if res_idx in bb_remodeled_residues:
			continue
		else:
			non_bb_remodeled_residues.append(res_idx)

	return bb_remodeled_residues, non_bb_remodeled_residues

def load_ss_lhl_res(start_struct, bb_remodeled_residues):
	## For getting designs' LHL residues, use get_lhl_residues(). 
	## This function is only used to get the LHL residues of the 
	## Starting Structure (which doesn't have linker_ids)
	## Returns a list of lists of LHL residue indices

	# Separate contingous LHLs
	lhl_residues = []
	prev_stop = 0
	for i in range(len(bb_remodeled_residues)-1):
		if bb_remodeled_residues[i] != bb_remodeled_residues[i+1] - 1:
			lhl_residues.append(bb_remodeled_residues[prev_stop:i+1])
			prev_stop = i + 1
	lhl_residues.append(bb_remodeled_residues[prev_stop:])

	# Determine helix residues with DSSP
	helix_residues = []
	for res_list in lhl_residues:
		helix_start_stop = get_helix_of_lhl_unit(start_struct, min(res_list), max(res_list))
		helix_residues.append([i for i in range(helix_start_stop[0], helix_start_stop[1]+1)])

	return lhl_residues, helix_residues


def get_design_ids_af_dirs(args):
	# Return a list of tuples, containing design IDs and their AF2/AF3 subdirectories.

	# Each af_file corresponds to an RO2 sequence, unless af_dir is in colabfold
	# format, in which case each af_file is a PDB, json, or png file.
	af_found = False

	if args.af_data_dirs:
		af_data_dirs = args.af_data_dirs
		af_found = True

	elif args.af_data_parents:
		af_data_dirs = []
		for parent_dir in args.af_data_parents:
			for subdir in os.listdir(parent_dir):
				af_data_dirs.append(os.path.join(parent_dir, subdir))
		af_found = True

	if args.include_designs_df != None:
		include_designs_df = pd.read_csv(args.include_designs_df)
		try:
			include_design_ids = include_designs_df['design_id'].tolist()
			if args.verbose:
				print(f'Number of included design IDs: {len(include_design_ids)}')
		except:
			print(f"design_id column not found in {args.include_designs_df}. Ignoring --include_designs_df")

	design_ids_af_dirs = []
	if af_found:
		for af_dir in af_data_dirs:
			if args.verbose:
				print(f'\naf_dir in get_design_ids_af_dirs(): {af_dir}')

			if args.is_af3_linux:
				# If AF3 Linux format, the design IDs are in subdirectories.
				# Each subdirectory is named by the design ID.
				for af_subdir in natsorted(os.listdir(af_dir)):
					if af_subdir == '.DS_Store': continue

					lucs_design_id = af_subdir.split('.')[0].split('_')[0]
					af_design_id = af_subdir.replace('mpnnseq', 'mpnnSeq') # AF3 linux format has design IDs in lowercase.
					curr_af_dir = os.path.join(af_dir, af_subdir)

					if (lucs_design_id, af_design_id, curr_af_dir) not in design_ids_af_dirs:
						design_ids_af_dirs.append((lucs_design_id, af_design_id, curr_af_dir))

					if args.test_stop != None:
						if len(design_ids_af_dirs) > args.test_stop:
							return design_ids_af_dirs
			else:
				for af_file in natsorted(os.listdir(af_dir)):
					# Ignore some of the Colabfold output files.
					if af_file in ['cite.bibtex', 'config.json', 'log.txt']: continue

					# Ignore any misc dirs containing files.
					if 'files' in af_file: continue

					# Colabfold format has file names beginning with the design ID
					# The alternative format is having each design's AF2/AF3 outputs in a
					# different folder, each folder named by design ID.
					if args.is_colabfold_format:
						lucs_design_id = str(af_file).split('.')[0].split('_')[0]
					else:
						lucs_design_id = str(af_file).split('.')[0]

					lucs_design_id = lucs_design_id.replace(args.af_prefix, '')

					# Add multiple LUCS and AF2/AF3 design IDs for each ProteinMPNN suffix.
					for suffix in args.pmpnn_suffixes:

						af_design_id = args.af_prefix+lucs_design_id+suffix

						if args.is_af3:
							# AF3 server converts jobnames to lowercase
							lucs_design_id = lucs_design_id.replace('mpnnseq', 'mpnnSeq')
							af_design_id = af_design_id.lower()

						if args.include_designs_df != None:
							if af_design_id not in include_design_ids:
								continue

						if (lucs_design_id, af_design_id, af_dir) not in design_ids_af_dirs:
							if args.is_af3_linux:
								design_ids_af_dirs.append((lucs_design_id, af_design_id, os.path.join(af_dir, af_design_id)))
							if args.is_colabfold_format and not args.is_af3: # AF3 models stored in af_dir/af_design_id/*
								design_ids_af_dirs.append((lucs_design_id, af_design_id, af_dir))
							else:
								design_ids_af_dirs.append((lucs_design_id, af_design_id, os.path.join(af_dir, af_design_id)))

					if args.test_stop != None:
						if len(design_ids_af_dirs) > args.test_stop:
							return design_ids_af_dirs

	# If no AF2/AF3 or prediction data is provided, use the args.lucs_data dir to load LUCS IDs
	else:
		for lucs_data_dir in args.lucs_data:
			for lucs_design_id in os.listdir(lucs_data_dir):
				if lucs_design_id == '.DS_Store': continue
				
				# Set the af_dir to the lucs_dir
				af_dir = os.path.join(lucs_data_dir, lucs_design_id)

				# Add multiple LUCS and AF2/AF3 design IDs for each ProteinMPNN suffix.
				for suffix in args.pmpnn_suffixes:

					af_design_id = lucs_design_id+suffix

					if (lucs_design_id, af_design_id, af_dir) not in design_ids_af_dirs:
						if args.is_colabfold_format:
							design_ids_af_dirs.append((lucs_design_id, af_design_id, af_dir))
						else:
							design_ids_af_dirs.append((lucs_design_id, af_design_id, os.path.join(af_dir, af_design_id)))

					if args.test_stop != None:
						if len(design_ids_af_dirs) > args.test_stop:
							return design_ids_af_dirs

	return design_ids_af_dirs

def get_lhl_residues(
	args,
	design_info_file, 
	seq_len,
	pose=None,
	use_dssp=False,
	assume_helix=True,
	bb_rem_all_helical=False):

	# Load the list of bb_remodeled_residues
	with open(design_info_file) as design_info_json:
		design_info = json.load(design_info_json)
		bb_remodeled_residues = design_info['bb_remodeled_residues']

	# Separate contingous LHLs
	lhl_residues = []
	prev_stop = 0
	for i in range(len(bb_remodeled_residues)-1):
		if bb_remodeled_residues[i] != bb_remodeled_residues[i+1] - 1:
			lhl_residues.append(bb_remodeled_residues[prev_stop:i+1])
			prev_stop = i + 1
	lhl_residues.append(bb_remodeled_residues[prev_stop:])

	linker_ids_found = False
	if not use_dssp:
		try:
			# Load the list of linker_ids
			with open(design_info_file) as design_info_json:
				design_info = json.load(design_info_json)
				linker_ids = design_info['linker_ids']
				linker_ids_found = True
		except:
			if args.verbose:
				print(f"design_info['linker_ids'] not found in {design_info_file}. Using DSSP to assign helix residues.")
			linker_ids_found = False

	helix_residues = []
	if bb_rem_all_helical:
		helix_residues = lhl_residues

	elif linker_ids_found and not use_dssp:
		# Load a list of bb_remodeled helix residue indices.
		for i in range(len(linker_ids)):
			l_d = linker_ids[i]
			# Use LUCS linker database nomenclature to get linker lengths
			front_loop_len = int(l_d['front_linker_id'].split('_non_redundant.json')[0][-1]) # 1-digit loop lengths
			back_loop_len = int(l_d['back_linker_id'].split('_non_redundant.json')[0][-1])

			if args.verbose:
				print(f'Linker IDs found with lengths {front_loop_len} and {back_loop_len}')

			# lhl_residues indices match those of linker_ids
			helix_residues.append(lhl_residues[i][front_loop_len:-1*back_loop_len])

	else:
		# Determine helix residues with DSSP
		for res_list in lhl_residues:
			if pose == None:
				print("If design_info['linker_ids'] is not found, a pose object must be passed to utils_degree_reshaped.get_lhl_residues()")
			else:
				helix_start_stop = get_helix_of_lhl_unit(pose, min(res_list), max(res_list))

			# If fewer than 4 helical residues were found, set the central residues (-5 from each end of the LHL) to be the helix,
			# as no LUCS loops are longer than 5 residues.
			if assume_helix and helix_start_stop[1] - helix_start_stop[0] < 4:
				print(f'Fewer than 4 helical residues found for {design_info_file}. Using helical residues {res_list[5:-5]}')
				h_start = res_list[0] + 5
				h_stop = res_list[-1] - 5
				helix_residues.append([i for i in range(h_start, h_stop+1)])

			else:
				# If no helical residues found, helix_start_stop == (-1, -1)
				if helix_start_stop[1] - helix_start_stop[0] == 0:
					print(f'WARNING: {helix_start_stop[1] - helix_start_stop[0]} helical residues found for {design_info_file}.')
					pass
				else:
					helix_residues.append([i for i in range(helix_start_stop[0], helix_start_stop[1]+1)])

	return lhl_residues, helix_residues

def get_design_to_ss_aligned_residues(args, fd_path):
	# Look in the design's folder for a {design_id}_align_residues.json file
	design_align_res_file = fd_path.replace('.pdb','_align_residues.json') 

	if not os.path.exists(design_align_res_file):
		print(f'{design_align_res_file} not found. Not using align_residues.json file for structure alignment.')
		return None
	
	if not os.path.exists(args.starting_structure_align_residues_file):
		print(f'{args.starting_structure_align_residues_file} not found. Not using align_residues.json file for structure alignment.')
		return None

	with open(design_align_res_file, 'r') as f:
		design_align_res = json.load(f)

	with open(args.starting_structure_align_residues_file, 'r') as f:
		ss_align_res = json.load(f)

	# Return a list of two lists, containing 1) the starting structure's and 2) the design's residues on which to align
	align_residues = [[],[]]
	for ss_res, des_res in zip(ss_align_res, design_align_res):
		if len(ss_res) > len(des_res):
			start_idx = (len(ss_res) - len(des_res)) // 2
			ss_res = ss_res[start_idx:start_idx+len(des_res)]
		elif len(des_res) > len(ss_res):
			start_idx = (len(des_res) - len(ss_res)) // 2
			des_res = des_res[start_idx:start_idx+len(ss_res)]

		align_residues[0].append(ss_res)
		align_residues[1].append(des_res)

	return align_residues

def convert_pdb_to_pose_resnums(pose, pdb_res_list: list[int], verbose: bool = False) -> list[int]:
	"""
	Convert PDB residue numbers to PyRosetta pose numbering.

	Maps residue indices from pdb file numbering to PyRosetta's internal
	pose numbering system, which may differ due to chain breaks or non-standard
	numbering in pdb files.

	Parameters
	----------
	pose : pyrosetta.Pose
		PyRosetta Pose object
	pdb_res_list : list of int
		List of residue numbers in PDB numbering
	verbose : bool, optional
		If True, print conversion details (default: False)

	Returns
	-------
	list of int
		List of residue numbers in PyRosetta pose numbering

	Notes
	-----
	- Residues that cannot be mapped (convert to 0) are excluded
	- Uses the chain ID from the first residue to determine the chain
	- Prints warnings for excluded residues if verbose=True

	Convert a list of PDB residue numbers to PyRosetta pose numbering
	"""
	# Ex. pose.pdb_info().pdb2pose('A', 601) # -> int
	# To convert from pose to PDB: pose.pdb_info().pose2pdb(24) # -> '47 A ' Res num, chain ID
	pose_res_list = []
	for pdb_res_num in pdb_res_list:

		chain = pose.pdb_info().pose2pdb(1).split()[1].strip()

		## Pose to PDB
		pose_res_num = int(pose.pdb_info().pdb2pose(chain, pdb_res_num))

		if pose_res_num == 0:
			if verbose:
				print(f'PDB res {pdb_res_num} converted to {pose_res_num}. Excluding this residue.')
		else:
			pose_res_list.append(pose_res_num)

	if verbose:
		print(f'Converted PDB residue numbering to PyRosetta Pose numbers:\n{pdb_res_list}\nto\n{pose_res_list}')

	return pose_res_list

def get_residue_list_from_pdb(pdb_path: str) -> list[int]:
	"""
	Extract all residue numbers from a pdb file.

	Parses ATOM records to find all unique residue numbers present in the
	pdb file.

	Parameters
	----------
	pdb_path : str
		Path to the pdb file

	Returns
	-------
	list of int
		Sorted list of unique residue numbers found in the pdb file

	Notes
	-----
	- Only parses ATOM records (not HETATM)
	- Maintains order of first occurrence
	- Residue numbers extracted from columns 23-26 of PDB format

	Find the residue numbers that exist in the pdb file
	"""
	resnums = []
	with open(pdb_path, 'r') as f:
		for line in f.readlines():
			if line.startswith('ATOM'):
				resnum = int(line[22:26])
				if resnum not in resnums:
					resnums.append(resnum)
	return resnums

def load_ss_lhl_res_from_file(helix_res_file, bb_remodeled_residues):
	"""
	Load starting structure helix residues from a provided 
	json file containing a list of lists of helix residues.
	"""
	# Separate contingous LHLs
	lhl_residues = []
	prev_stop = 0
	for i in range(len(bb_remodeled_residues)-1):
		if bb_remodeled_residues[i] != bb_remodeled_residues[i+1] - 1:
			lhl_residues.append(bb_remodeled_residues[prev_stop:i+1])
			prev_stop = i + 1
	lhl_residues.append(bb_remodeled_residues[prev_stop:])

	# Load helix residues from the json file
	with open(helix_res_file, 'r') as f:
		helix_residues = json.load(f)
	
	return lhl_residues, helix_residues



### Adding this function to find the longest common helices, in
### order to handle proteins of different lengths (and with non-matching
### bb-remodeled residue indices).
def find_longest_common_H_substring(s1, s2):
    """
    Find the longest common substring of 'H's between two strings.

    Args:
        s1 (str): First string containing 'H', 'E', and 'D'.
        s2 (str): Second string containing 'H', 'E', and 'D'.

    Returns:
        tuple: Indices of the start and end positions of the substring in s1 and s2.
               Format: ((s1_start, s1_end), (s2_start, s2_end))
    """
    def get_H_substrings_indices(s):
        """Get all substrings consisting of 'H' and their start and end indices."""
        substrings = []
        start = None
        for i, char in enumerate(s):
            if char == 'H':
                if start is None:  # Mark start of 'H' substring
                    start = i
            else:
                if start is not None:  # End of 'H' substring
                    substrings.append((start, i))  # Exclude current non-H character
                    start = None
        if start is not None:  # Handle case where string ends with 'H'
            substrings.append((start, len(s)))
        return substrings

    # Extract all 'H' substrings with their indices
    s1_H_substrings = get_H_substrings_indices(s1)
    s2_H_substrings = get_H_substrings_indices(s2)

    # Compare all 'H' substrings to find the longest common one
    longest_common = None
    longest_length = 0
    indices_in_s1 = (-1, -1)
    indices_in_s2 = (-1, -1)

    for s1_start, s1_end in s1_H_substrings:
        s1_sub = s1[s1_start:s1_end]
        for s2_start, s2_end in s2_H_substrings:
            s2_sub = s2[s2_start:s2_end]
            # Find the common prefix of these substrings
            common_length = 0
            while (common_length < len(s1_sub) and
                   common_length < len(s2_sub) and
                   s1_sub[common_length] == 'H' and
                   s2_sub[common_length] == 'H'):
                common_length += 1

            # Update the longest common substring if needed
            if common_length > longest_length:
                longest_length = common_length
                longest_common = s1_sub[:common_length]
                indices_in_s1 = (s1_start, s1_start + common_length)
                indices_in_s2 = (s2_start, s2_start + common_length)

    # Return the results
    if longest_length > 0:
        return indices_in_s1, indices_in_s2
    else:
        print(f"No common 'H' substring found between the two strings:\n{s1_H_substrings}\n{s2_H_substrings}")
        print(f"Returning None for shared H substrings in utils.find_longest_common_H_substring()")
        return None, None  # No common 'H' substring found



#####
### Steph's Helix RMSD functions (adapted from Dom and Rob)
### Code modified by Ben Orr (adding loops, changing variable names, 
### and adjusting for different-length input poses)
#####
# DWG RMSD functions
def steph_align_by_residues(input_pose, ref_pose, input_align_res, ref_align_res):
	"""
	Adapted from @dwgrisingher
	Aligns two poses by residue selector over the non-bb remodeled residues

	align_res is a list of strings of residue ranges, e.g. ['1-20', '30-40']

	In cases where a selected residue is not found in the input_pose, trim
	the C-terminal residues of the last input_align_res residue range until
	all residues are found in the input_pose. If the last contiguous segment
	of the input_align_res is not found in the pose, then throw an error and exit.
	"""

	# Trim the residue ranges so that each range is the same length
	for input_res_range, ref_res_range in zip(input_align_res, ref_align_res):
		len_input_res_range = int(input_res_range.split('-')[1]) - int(input_res_range.split('-')[0]) + 1
		len_ref_res_range = int(ref_res_range.split('-')[1]) - int(ref_res_range.split('-')[0]) + 1
		if len_input_res_range > len_ref_res_range:
			input_align_res[input_align_res.index(input_res_range)] = f"{input_res_range.split('-')[0]}-{int(input_res_range.split('-')[0]) + len_ref_res_range - 1}"
		elif len_input_res_range < len_ref_res_range:
			ref_align_res[ref_align_res.index(ref_res_range)] = f"{ref_res_range.split('-')[0]}-{int(ref_res_range.split('-')[0]) + len_input_res_range - 1}"

	# Select residues in input pose
	# Expects 'res_i-res_j' strings as input
	input_res = OrResidueSelector()
	for input_res_range in input_align_res:
		input_res.add_residue_selector(ResidueIndexSelector(input_res_range))

	# Select residues in reference pose
	ref_res = OrResidueSelector()
	for ref_res_range in ref_align_res:
		ref_res.add_residue_selector(ResidueIndexSelector(ref_res_range))

	abr = AlignByResidueSelectorMover()
	abr.query_selector(input_res)
	abr.reference_pose(ref_pose)
	abr.reference_selector(ref_res)
	
	abr.apply(input_pose)

	return input_pose, ref_pose

def steph_calculate_RMSDs_on_beta_only_helix(input_pose, ref_pose, input_rmsd_res, ref_rmsd_res):
	"""
	Adapted from @dwgrisingher
	Takes two already-aligned poses and returns bb RMSDs

	Inputs:
		input_pose and ref_pose: PyRosetta pose objects
		input_rmsd_res and ref_rmsd_res: list of strings containing residue ranges, such as ["31-51", "58-71"]
	
	Returns:
		helix_rmsds: list of floats with helix RMSDs corresponding to reshaped LHL indices.
	
	"""
	skip_idxs = []

	# Select residues in input pose
	input_res_selectors = []
	for i, input_res_range in enumerate(input_rmsd_res):
		if input_res_range == None:
			input_res_selectors.append(None)
			skip_idxs.append(i)
		else:
			input_res_selectors.append(ResidueIndexSelector(input_res_range))

	# Select residues in reference pose
	ref_res_selectors = []
	for ref_res_range in ref_rmsd_res:
		if ref_res_range == None:
			ref_res_selectors.append(None)
			skip_idxs.append(i)
		else:
			ref_res_selectors.append(ResidueIndexSelector(ref_res_range))

	# Set up the RMSDMetric
	rmsd_metric = RMSDMetric()
	rmsd_metric.set_run_superimpose(False)
	rmsd_metric.set_comparison_pose(input_pose)
	rmsd_metric.set_rmsd_type(pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_heavy_including_O)

	# Calculate Helix RMSDs
	helix_rmsds = []
	for i, (input_res, ref_res) in enumerate(zip(input_res_selectors, ref_res_selectors)):
		if i in skip_idxs:
			helix_rmsds.append(np.nan)
			print(f"Skipping RMSD calculation for helix {i} due to rmsd_residues==None in utils.steph_calculate_RMSDs_on_beta_only_helix()")
		else:
			rmsd_metric.set_residue_selector(ref_res)
			rmsd_metric.set_residue_selector_reference(input_res)
			helix_rmsds.append(rmsd_metric.calculate(ref_pose))

	return helix_rmsds

# SEC RMSD functions
# function to get dssp of remodeled residues
# get shared residue numbers of helices and append for calculating rmsd
# can use non-bb remodeled for aligning
def steph_get_shared_helical_resn(input_pose, 
								  ref_pose, 
								  input_bb_rem_res, 
								  ref_bb_rem_res,
								  verbose=True):
	'''
	Return the residue indices for two poses (pyrosetta 1-indexed residue numbers)
	corresponding to their shared helix residues.

	Parameters:
	input_pose (pose): pose 1
	ref_pose (pose): pose 2
	input_bb_rem_res (list of lists of ints): list of start and stop residue numbers of first remodeled helix (pose numbering - 1-indexed)
	ref_bb_rem_res (list of lists of ints): list of start and stop residue numbers of second remodeled helix (pose numbering - 1-indexed)

	Returns:
	input_shared_helix_res (list of list of ints): list of residue numbers in the input pose of shared remodeled helix residues
	ref_shared_helix_res (list of list of ints): list of residue numbers in the ref pose of shared remodeled helix residues
	'''

	#get dssp of rosetta pose
	dssp_mover = DsspMover()
	dssp_mover.apply(input_pose)
	input_ss = input_pose.secstruct()

	#get dssp of af pose
	dssp_mover = DsspMover()
	dssp_mover.apply(ref_pose)
	ref_ss = ref_pose.secstruct()

	#get ss assignments for remodeled resns
	# Index the 0-indexed DSSP string with 1-indexed pyrosetta pose numbers
	input_bb_rem_ss = []
	for bb_rem_res in input_bb_rem_res:
		input_bb_rem_ss.append(input_ss[bb_rem_res[0]-1: bb_rem_res[-1]])

	ref_bb_rem_ss = []
	for bb_rem_res in ref_bb_rem_res:
		ref_bb_rem_ss.append(ref_ss[bb_rem_res[0]-1: bb_rem_res[-1]])

	### Rewriting how shared helix residues are determined, to handle cases
	### where proteins of different lengths are being compared.
	### Now, this will find the longest common helix.
	input_shared_helix_res = []
	ref_shared_helix_res = []
	for input_str, ref_str, input_bb_rem_res_contig, ref_bb_rem_res_contig in zip(input_bb_rem_ss, ref_bb_rem_ss, input_bb_rem_res, ref_bb_rem_res):
		input_shared_helix, ref_shared_helix = find_longest_common_H_substring(input_str, ref_str)
		if input_shared_helix == None:
			input_shared_helix_res.append([])
			ref_shared_helix_res.append([])
			continue
		# Convert 0-indexed DSSP indices to 1-indexed Pose indices
		# find_longest_common_H_substring returns the indices in the input substrings, which
		# need to be converted to the full input_pose and ref_pose residue indices (+input_bb_rem_res_contig[0])
		# This only works if the bb-remodeled residue contig is not missing residues.
		input_shared_helix_res.append([i+1+input_bb_rem_res_contig[0] for i in range(input_shared_helix[0], input_shared_helix[-1]+1)])
		ref_shared_helix_res.append([i+1+ref_bb_rem_res_contig[0] for i in range(ref_shared_helix[0], ref_shared_helix[-1]+1)])

	if verbose:
		print("Longest Common Helices in utils.steph_get_shared_helical_resn():")
		print("input_bb_rem_res:")
		print(input_bb_rem_res)
		print("input_ss")
		print(input_ss)
		print("ref_bb_rem_res")
		print(ref_bb_rem_res)
		print("ref_ss")
		print(ref_ss)
		print("input_bb_rem_ss")
		print(input_bb_rem_ss)
		print("ref_bb_rem_ss")
		print(ref_bb_rem_ss)
		print("input_shared_helix_res")
		print(input_shared_helix_res)
		print("ref_shared_helix_res")
		print(ref_shared_helix_res)

	return input_shared_helix_res, ref_shared_helix_res

#####
### End Steph's Helix RMSD functions
#####


def split_contiguous_residues(res_list: list[int]) -> list[list[int]]:
	"""
	Split a list of contiguous residue numbers into separate lists.

	Divides a list of residue numbers into sublists where each sublist
	contains only consecutive (contiguous) residue numbers.

	Parameters
	----------
	res_list : list of int
		List of residue numbers (can be non-contiguous)

	Returns
	-------
	list of list of int
		List of sublists, where each sublist contains contiguous residues

	Examples
	--------
	>>> split_contiguous_residues([1,2,3,5,6,7,8,10])
	[[1,2,3], [5,6,7,8], [10]]

	>>> split_contiguous_residues([1,2,3,4,5])
	[[1,2,3,4,5]]

	Notes
	-----
	- Assumes input list is sorted
	- Useful for identifying separate secondary structure elements

	Example:
	[1,2,3,5,6,7,8,10] -> [[1,2,3], [5,6,7,8], [10]]
	"""
	split_res_list = []
	prev_res = res_list[0]
	contig_res = [prev_res]
	for res in res_list[1:]:
		if res == prev_res + 1:
			contig_res.append(res)
		else:
			split_res_list.append(contig_res)
			contig_res = [res]
		prev_res = res
	split_res_list.append(contig_res)

	return split_res_list
