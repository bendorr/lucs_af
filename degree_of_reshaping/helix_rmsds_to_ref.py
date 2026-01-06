<<<<<<< HEAD
import os
import json
import argparse
import utils_degree_reshaped as utils
=======
"""
Computes helix RMSDs between protein designs and a reference structure.

Author: Ben Orr
Date: 3.20.25

This module computes Root Mean Square Deviations (RMSDs) for helical regions between
protein designs and a reference structure. It supports both experimental designs and
AlphaFold2/3 predictions, with multiple alignment and RMSD calculation methods.

Usage Examples
--------------
Basic usage with design structures::

    python helix_rmsds_to_ref.py \\
        --pdb_dir /path/to/designs/ \\
        --design_info_dir /path/to/design_info/ \\
        --outfile results/helix_rmsds.csv \\
        --reference_structure_path /path/to/reference.pdb \\
        --reference_structure_design_info /path/to/ref_design_info.json

With AlphaFold2 predictions::

    python helix_rmsds_to_ref.py \\
        --analyze_af2 \\
        --af2_data_parents /path/to/af2_predictions/ \\
        --design_info_dir /path/to/design_info/ \\
        --outfile results/af2_helix_rmsds.csv \\
        --reference_structure_path /path/to/reference.pdb \\
        --reference_structure_design_info /path/to/ref_design_info.json

With parallel processing::

    python helix_rmsds_to_ref.py \\
        --pdb_dir /path/to/designs/ \\
        --design_info_dir /path/to/design_info/ \\
        --outfile results/helix_rmsds.csv \\
        --num_tasks 10 \\
        --task_id 1

Parameters
----------
Input/Output:
    --pdb_dir : list of str
        Directories containing design PDB files in subdirectories named by design ID
    --design_info_dir : list of str
        Directories containing design_info.json files
    --outfile : str
        Path to output CSV file for results
    --design_id_suffix : str, optional
        Suffix for PDB filenames (default: '')
    --include_designs_df : str, optional
        CSV file with 'design_id' column to filter designs

AlphaFold2/3 Options:
    --analyze_af2 : flag
        Analyze AlphaFold predictions instead of design PDBs
    --af2_data_parents : list of str
        Parent directories containing AF2 prediction subdirectories
    --is_colabfold_format : flag
        AF2 data is in ColabFold format
    --af2_suffixes : list of str
        Suffixes for AF2 prediction filenames
    --is_af3 : flag
        Predictions are from AlphaFold3

Alignment/RMSD Options:
    --reference_structure_path : str
        Path to reference structure PDB
    --reference_structure_design_info : str
        Path to reference structure design info JSON
    --use_dssp : flag
        Use DSSP to identify helical residues
    --use_align_residues_file : flag
        Use align_residues.json for structure alignment

Parallelization:
    --num_tasks : int
        Number of parallel tasks (default: 1)
    --task_id : int
        Current task ID (1-indexed)

Output
------
CSV file containing:
    - design_path: Path to design PDB file
    - design_id: Design identifier
    - ben_helix_rmsd: Overall helix RMSD using custom alignment
    - ben_indiv_helix{i}_rmsd: Individual helix RMSD for helix i
    - indiv_helix{i}_rmsd2: Individual helix RMSD using alternative method
    - indiv_common_helix{i}_rmsd: RMSD for shared helical regions

Notes
-----
- Requires PyRosetta for structure manipulation and RMSD calculations
- Supports multiple alignment strategies (non-remodeled residues, custom alignment files)
- Calculates RMSDs using both custom and standard methods for validation
- Can process large batches with SGE-style parallelization
"""

import os
import json
import argparse
from typing import List, Tuple, Dict, Optional, Any
>>>>>>> be02a1e (lucs_af refactor and cleanup)
import numpy as np
import pandas as pd
from natsort import natsorted

<<<<<<< HEAD
from pyrosetta import init, pose_from_file
# init pyrosetta
=======
import utils_degree_reshaped as utils
from pyrosetta import init, pose_from_file
from pyrosetta.rosetta.core.pose import Pose

# Initialize PyRosetta with muted output
>>>>>>> be02a1e (lucs_af refactor and cleanup)
init(options='-mute all')

#####

<<<<<<< HEAD
def check_completed_files(args):
=======
def check_completed_files(args: argparse.Namespace) -> bool:
    """
    Check if output file for current task already exists.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing num_tasks, task_id, and outfile

    Returns
    -------
    bool
        True if output file exists, False otherwise

    Notes
    -----
    When parallelizing with multiple tasks, output files are named with
    task identifiers: outfile_task{task_id}_of_{num_tasks}.csv
    """
>>>>>>> be02a1e (lucs_af refactor and cleanup)
    if args.num_tasks != 1:
        fname_append = f'_task{args.task_id}_of_{args.num_tasks}'
    else:
        fname_append = ''
    df_outpath = os.path.join(args.outfile.replace('.csv', f'{fname_append}.csv'))
    if os.path.exists(df_outpath):
        return True
    return False

<<<<<<< HEAD
def get_design_paths(args):
=======
def get_design_paths(args: argparse.Namespace) -> List[Tuple[str, str]]:
    """
    Collect paths to design PDB files from specified directories.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - pdb_dir: List of directories with design subdirectories
        - design_id_suffix: Optional suffix for PDB filenames
        - include_designs_df: Optional CSV file to filter designs

    Returns
    -------
    List[Tuple[str, str]]
        List of (design_path, design_id) tuples, sorted alphabetically

    Notes
    -----
    Expects directory structure: pdb_dir/design_id/design_id{suffix}.pdb
    If include_designs_df is provided, only designs in the 'design_id' column
    will be included.
    """
>>>>>>> be02a1e (lucs_af refactor and cleanup)
    design_paths_ids = []
    if args.include_designs_df != None:
        include_designs_df = pd.read_csv(args.include_designs_df)
        include_design_ids = include_designs_df['design_id'].tolist()

    for data_dir in args.pdb_dir:
        for subdir in os.listdir(data_dir):
            if args.include_designs_df != None:
                if subdir not in include_design_ids:
                    continue
            design_path = os.path.join(data_dir, subdir, f"{subdir}{args.design_id_suffix}.pdb")
            if os.path.exists(design_path):
                design_paths_ids.append((design_path, f"{subdir}{args.design_id_suffix}"))

    # Sort the design paths
    design_paths_ids.sort()

    return design_paths_ids

<<<<<<< HEAD
def get_af2_design_paths(args):
    # Return a list of tuples, containing design IDs and their AF2 subdirectories.

    # Each af_file corresponds to an RO2 sequence, unless af_dir is in colabfold
    # format, in which case each af_file is a PDB, json, or png file.
=======
def get_af2_design_paths(args: argparse.Namespace) -> List[Tuple[str, str]]:
    """
    Collect paths to AlphaFold2/3 prediction files from specified directories.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - af2_data_parents: Parent directories containing AF2 predictions
        - include_designs_df: Optional CSV file to filter designs
        - is_colabfold_format: Whether data is in ColabFold format
        - split_af2_fnames_at: String to split filenames for design IDs
        - af_prefix: Prefix for AF2/3 prediction filenames
        - af2_suffixes: List of suffixes for AF2 predictions
        - is_af3: Whether predictions are from AlphaFold3
        - test_stop: Optional limit on number of designs to process
        - verbose: Print detailed output

    Returns
    -------
    List[Tuple[str, str]]
        List of (af_prediction_path, af_design_id) tuples

    Notes
    -----
    - Each af_file corresponds to an RO2 sequence, unless af_dir is in ColabFold
      format, in which case each af_file is a PDB, json, or png file.
    - Supports both ColabFold format (all files in one directory) and standard
      format (separate subdirectory per design)
    - Only processes PDB files (change for AF3 .cif files)
    """
>>>>>>> be02a1e (lucs_af refactor and cleanup)
    af2_found = False

    if args.af2_data_parents:
        af2_data_dirs = []
        for parent_dir in args.af2_data_parents:
            for subdir in os.listdir(parent_dir):
                af2_data_dirs.append(os.path.join(parent_dir, subdir))
        af2_found = True

    if args.include_designs_df != None:
        include_designs_df = pd.read_csv(args.include_designs_df)
        try:
            include_design_ids = include_designs_df['design_id'].tolist()
            if args.verbose:
                print(f'Number of included design IDs: {len(include_design_ids)}')
        except:
            print(f"design_id column not found in {args.include_designs_df}. Ignoring --include_designs_df")

    design_paths_ids = []
    if af2_found:
        for af_dir in af2_data_dirs:
            if args.verbose:
                print(f'\naf_dir in get_af2_design_paths(): {af_dir}')

            for af_file in natsorted(os.listdir(af_dir)):
                # Ignore some of the Colabfold output files.
                if af_file in ['cite.bibtex', 'config.json', 'log.txt']: continue

                # Ignore misc dirs containing files.
                if 'files' in af_file: continue

                # Only process PDB files.
                # Change this if processing AF3 .cif files.
                if '.pdb' not in af_file: continue

                # if args.verbose:
                # 	print(f'\naf_file in get_design_ids_af_dirs(): {af_file}')

                # Colabfold format has file names beginning with the design ID
                # The alternative format is having each design's AF2 outputs in a
                # different folder, each folder named by design ID.
                if args.is_colabfold_format:
                    design_id = str(af_file).split('.')[0].split('_')[0]
                else:
                    design_id = str(af_file).split('.')[0]

                print(f"Design ID 1: {design_id}")

                # Split the design_id at args.split_af2_fnames_at
                if args.split_af2_fnames_at != '':
                    design_id = design_id.split(args.split_af2_fnames_at)[0]

                print(f"Design ID 2: {design_id}")
                
                design_id = design_id.replace(args.af_prefix, '')

                print(f"Design ID 3: {design_id}")

                # Add multiple AF2 design IDs (one for each af2_suffix)
                for suffix in args.af2_suffixes:

                    af2_design_id = args.af_prefix+design_id+suffix

                    print(f"AF2 Design ID 1: {af2_design_id}")

                    # if args.is_af3:
                    #     # AF3 server converts jobnames to lowercase
                    #     lucs_design_id = lucs_design_id.replace('mpnnseq', 'mpnnSeq')
                    #     af2_design_id = af2_design_id.lower()

                    if args.include_designs_df != None:
                        if af2_design_id not in include_design_ids:
                            # if args.verbose:
                            # 	print(f'{af2_design_id} not in include_design_ids. Skipping.')
                            continue
                    
                    if args.is_colabfold_format and not args.is_af3: # AF3 models stored in af_dir/af_design_id/*
                        af_path = os.path.join(af_dir, af_file)
                    else:
                        af_path = os.path.join(af_dir, af2_design_id, af_file)

                    if (af_path, af2_design_id) not in design_paths_ids:
                        design_paths_ids.append((af_path, af2_design_id))

                if args.test_stop != None:
                    if len(design_paths_ids) > args.test_stop:
                        return design_paths_ids
<<<<<<< HEAD
                    
    return design_paths_ids

def get_design_info_path(args, design_id):
    # args.design_info_dir is a list of dirs. Find the design's
    # subdir in one of these dirs.
=======

    return design_paths_ids

def get_design_info_path(args: argparse.Namespace, design_id: str) -> str:
    """
    Find the design_info.json file path for a given design ID.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing design_info_dir list
    design_id : str
        Design identifier to search for

    Returns
    -------
    str
        Path to the design_info.json file

    Raises
    ------
    FileNotFoundError
        If design_info file cannot be found for the design_id

    Notes
    -----
    - args.design_info_dir is a list of directories. This function finds the design's
      subdirectory in one of these directories.
    - Tries multiple filename patterns: {design_id}_design_info.json and design_info.json
    """
>>>>>>> be02a1e (lucs_af refactor and cleanup)
    for design_info_dir in args.design_info_dir:
        for subdir in os.listdir(design_info_dir):
            if subdir == design_id:
                design_info_path = os.path.join(design_info_dir, subdir, f'{subdir}_design_info.json')
                fnames = [design_info_path,
                        design_info_path.replace(f'{subdir}_','')]
                for fname in fnames:
                    if os.path.exists(fname):
                        return fname
                raise FileNotFoundError(f"Design info file not found for {design_id}")

<<<<<<< HEAD
def get_align_residues(args, design_path1, design_path2):
    # Look in the design's folder for a {design_id}_align_residues.json file
    design1_align_res_file = design_path1.replace('.pdb','_align_residues.json') 

    # Look in the design's folder for a {design_id}_align_residues.json file
    design2_align_res_file = design_path2.replace('.pdb','_align_residues.json') 
=======
def get_align_residues(args: argparse.Namespace, design_path1: str, design_path2: str) -> Optional[List[List[List[int]]]]:
    """
    Load alignment residues from JSON files for two structures.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments (not currently used in function body)
    design_path1 : str
        Path to first design PDB file
    design_path2 : str
        Path to second design PDB file

    Returns
    -------
    Optional[List[List[List[int]]]]
        List of two lists: [[ref_align_residues], [design_align_residues]]
        Each inner list contains lists of contiguous residue ranges for alignment.
        Returns None if either alignment file is not found.

    Notes
    -----
    - Looks for {design_id}_align_residues.json files in the same directory as PDB files
    - If designs have different lengths, centers the alignment on the shorter design
    - Residue numbers are expected to be PDB-numbered (will need conversion to Pose numbering)
    """
    # Look in the design's folder for a {design_id}_align_residues.json file
    design1_align_res_file = design_path1.replace('.pdb', '_align_residues.json')

    # Look in the design's folder for a {design_id}_align_residues.json file
    design2_align_res_file = design_path2.replace('.pdb', '_align_residues.json') 
>>>>>>> be02a1e (lucs_af refactor and cleanup)

    # Check if the files exist
    if not os.path.exists(design1_align_res_file):
        print(f'{design1_align_res_file} not found. Not using align_residues.json file for structure alignment.')
        return None
    if not os.path.exists(design2_align_res_file):
        print(f'{design2_align_res_file} not found. Not using align_residues.json file for structure alignment.')
        return None

    with open(design1_align_res_file, 'r') as f:
        design1_align_res = json.load(f)

    with open(design2_align_res_file, 'r') as f:
        design2_align_res = json.load(f)

    # Return a list of two lists, containing 1) the reference structure's and 2) the design's residues on which to align
<<<<<<< HEAD
    align_residues = [[],[]]
=======
    align_residues = [[], []]
>>>>>>> be02a1e (lucs_af refactor and cleanup)
    for des1_res, des2_res in zip(design1_align_res, design2_align_res):
        if len(des1_res) > len(des2_res):
            start_idx = (len(des1_res) - len(des2_res)) // 2
            des1_res = des1_res[start_idx:start_idx+len(des2_res)]
        elif len(des2_res) > len(des1_res):
            start_idx = (len(des2_res) - len(des1_res)) // 2
            des2_res = des2_res[start_idx:start_idx+len(des1_res)]

        align_residues[0].append(des1_res)
        align_residues[1].append(des2_res)

    return align_residues

<<<<<<< HEAD
def calculate_helix_rmsds_to_ref(args, design_paths_ids):
    """
    Calculate the helix RMSDs between the designs and the reference structure.
=======
def calculate_helix_rmsds_to_ref(args: argparse.Namespace, design_paths_ids: List[Tuple[str, str]]) -> Dict[str, List[Any]]:
    """
    Calculate helix RMSDs between designs and reference structure.

    This is the main processing function that:
    1. Loads the reference structure and identifies its helix residues
    2. For each design:
       - Loads the structure and identifies helix residues
       - Aligns design to reference
       - Calculates multiple RMSD metrics for helices
    3. Returns all RMSD results in a dictionary

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - reference_structure_path: Path to reference PDB
        - reference_structure_design_info: Path to reference design info JSON
        - reference_structure_helix_residues_file: Optional helix residues file
        - use_dssp: Use DSSP for secondary structure assignment
        - assume_helix: Assume all residues are helical
        - bb_rem_all_helical: Consider all helical residues as backbone-remodeled
        - use_align_residues_file: Use custom alignment residues
        - num_tasks: Number of parallel tasks
        - task_id: Current task ID
        - test_stop: Optional limit on number of calculations
        - verbose: Print detailed output
    design_paths_ids : List[Tuple[str, str]]
        List of (design_path, design_id) tuples to process

    Returns
    -------
    Dict[str, List[Any]]
        Dictionary containing RMSD results with keys:
        - 'design_path': List of design file paths
        - 'design_id': List of design identifiers
        - 'ben_helix_rmsd': Overall helix RMSD using custom alignment
        - 'ben_indiv_helix{i}_rmsd': Individual helix RMSD for helix i
        - 'indiv_helix{i}_rmsd2': Individual helix RMSD using alternative method
        - 'indiv_common_helix{i}_rmsd': RMSD for shared helical regions

    Notes
    -----
    - Uses multiple RMSD calculation methods for validation:
      * Custom Ben's alignment and RMSD calculation
      * Steph's alignment with standard PyRosetta RMSD
      * Common helix RMSD for shared helical regions
    - Automatically handles PDB to PyRosetta residue numbering conversion
    - Supports parallelization via task_id and num_tasks parameters
>>>>>>> be02a1e (lucs_af refactor and cleanup)
    """
    # Define a results_dict to store the results
    ret = {
        'design_path': [],
        'design_id': [],
        'ben_helix_rmsd': []
    }

    task_idx = 0
    completed_rmsd_count = 0
    force_stop = False

    #####
    ### Load the reference structure and its helix residues
    #####

    # Load the reference structure
    ref_path = args.reference_structure_path
    ref_pose = pose_from_file(ref_path)

    # Get PDB-numbered non-backbone remodeled residues
    pdb_ref_bb_remodeled_residues, pdb_ref_non_bb_remodeled_residues = \
        utils.get_remodeled_residues_from_design_info(
<<<<<<< HEAD
            pdb_file = ref_path,
            design_info_file = args.reference_structure_design_info)
    
    if args.reference_structure_helix_residues_file:
        pdb_ref_lhl_residues, pdb_ref_helix_residues = utils.load_ss_lhl_res_from_file(
                                                            args.reference_structure_helix_residues_file, 
                                                            pdb_ref_bb_remodeled_residues)
    else:
        # Find the helix residues of the reference structure. This list has len(num_lhls)
        pdb_ref_lhl_residues, pdb_ref_helix_residues = utils.get_lhl_residues(
                design_info_file = args.reference_structure_design_info,
                seq_len = len(ref_pose.sequence()),
                pose=ref_pose,
                use_dssp=args.use_dssp,
                assume_helix=args.assume_helix,
                bb_rem_all_helical=args.bb_rem_all_helical)
=======
            pdb_file=ref_path,
            design_info_file=args.reference_structure_design_info)
    
    if args.reference_structure_helix_residues_file:
        pdb_ref_lhl_residues, pdb_ref_helix_residues = utils.load_ss_lhl_res_from_file(
            args.reference_structure_helix_residues_file,
            pdb_ref_bb_remodeled_residues)
    else:
        # Find the helix residues of the reference structure. This list has len(num_lhls)
        pdb_ref_lhl_residues, pdb_ref_helix_residues = utils.get_lhl_residues(
            design_info_file=args.reference_structure_design_info,
            seq_len=len(ref_pose.sequence()),
            pose=ref_pose,
            use_dssp=args.use_dssp,
            assume_helix=args.assume_helix,
            bb_rem_all_helical=args.bb_rem_all_helical)
>>>>>>> be02a1e (lucs_af refactor and cleanup)
    
    if args.verbose:
        print(f'{args.reference_structure_design_info} LHL residues:')
        for res_list in pdb_ref_lhl_residues:
            print(res_list)
        print(f'{args.reference_structure_design_info} Helix residues:')
        for res_list in pdb_ref_helix_residues:
            print(res_list)

    # Convert between PDB and PyRosetta residue numberings for the reference structure
    print(f'Converting PDB to Rosetta numbering for {ref_path}:')
    ref_bb_remodeled_residues = utils.convert_pdb_to_pose_resnums(ref_pose, pdb_ref_bb_remodeled_residues, verbose=args.verbose)
    ref_non_bb_remodeled_residues = utils.convert_pdb_to_pose_resnums(ref_pose, pdb_ref_non_bb_remodeled_residues, verbose=args.verbose)
    ref_lhl_residues = []
    for i in range(len(pdb_ref_lhl_residues)):
        ref_lhl_residues.append([])
        ref_lhl_residues[i] = utils.convert_pdb_to_pose_resnums(ref_pose, pdb_ref_lhl_residues[i], verbose=args.verbose)
    ref_helix_residues = []
    for i in range(len(pdb_ref_helix_residues)):
        ref_helix_residues.append([])
        ref_helix_residues[i] = utils.convert_pdb_to_pose_resnums(ref_pose, pdb_ref_helix_residues[i], verbose=args.verbose)

    # Load strings containing residue ranges for PyRosetta
    ref_non_bb_remodeled_residues_string_list, ref_bb_remodeled_residues_string_list = \
        utils.load_pyrosetta_strings(ref_path, ref_lhl_residues)
    
    if args.verbose:
        print(f'{ref_path} bb_rem and non-bb_rem residues:')
        print(ref_non_bb_remodeled_residues_string_list)
        print(ref_bb_remodeled_residues_string_list)

    #####
    ### Finish loading reference structure
    #####

    for design_idx, (design_path, design_id) in enumerate(design_paths_ids):
        if force_stop:
            break

        task_idx += 1
        if task_idx % args.num_tasks != (args.task_id-1):
            continue

        print(f"\n\nCalculating RMSD index {task_idx}\nTask {args.task_id} of \
{args.num_tasks}\nDesign idx {design_idx} of {len(design_paths_ids)}\n{design_path} to {ref_path}\n\n")

        # Append the design path and design ID to the results dict
        ret['design_path'].append(design_path)
        ret['design_id'].append(design_id)

        # Load the design structure
        pose = pose_from_file(design_path)

        # Get design_info_path
        design_info_path = get_design_info_path(args, design_id)

        if args.verbose:
            print(f'{design_id} Design Info Path: {design_info_path}')

        # Get PDB-numbered non-backbone remodeled residues
        pdb_pose_bb_remodeled_residues, pdb_pose_non_bb_remodeled_residues = \
<<<<<<< HEAD
			utils.get_remodeled_residues_from_design_info(
				pdb_file = design_path,
				design_info_file = design_info_path)
=======
            utils.get_remodeled_residues_from_design_info(
                pdb_file=design_path,
                design_info_file=design_info_path)
>>>>>>> be02a1e (lucs_af refactor and cleanup)
        
        if args.verbose:
            print(f'{design_path} bb_remodeled_residues (PDB-numbered):')
            print(f"{pdb_pose_bb_remodeled_residues}")
            print(f'{design_path} non_bb_remodeled_residues (PDB-numbered):')
            print(f"{pdb_pose_non_bb_remodeled_residues}\n")

        # Find the helix residues of design1. This list has len(num_lhls)
        pdb_pose_lhl_residues, pdb_pose_helix_residues = utils.get_lhl_residues(
<<<<<<< HEAD
                design_info_file = design_info_path,
                seq_len = len(pose.sequence()),
                pose=pose,
                use_dssp=args.use_dssp,
                assume_helix=args.assume_helix,
                bb_rem_all_helical=args.bb_rem_all_helical)
=======
            design_info_file=design_info_path,
            seq_len=len(pose.sequence()),
            pose=pose,
            use_dssp=args.use_dssp,
            assume_helix=args.assume_helix,
            bb_rem_all_helical=args.bb_rem_all_helical)
>>>>>>> be02a1e (lucs_af refactor and cleanup)
        
        if args.verbose:
            print(f'{design_path} LHL residues:')
            for res_list in pdb_pose_lhl_residues:
                print(res_list)
            print(f'{design_path} Helix residues:')
            for res_list in pdb_pose_helix_residues:
                print(res_list)

        # Convert between PDB and PyRosetta residue numberings for design
        print(f'Converting PDB to Rosetta numbering for {design_path}:')
        pose_bb_remodeled_residues = utils.convert_pdb_to_pose_resnums(pose, pdb_pose_bb_remodeled_residues, verbose=args.verbose)
        pose_non_bb_remodeled_residues = utils.convert_pdb_to_pose_resnums(pose, pdb_pose_non_bb_remodeled_residues, verbose=args.verbose)
        pose_lhl_residues = []
        for i in range(len(pdb_pose_lhl_residues)):
            pose_lhl_residues.append([])
            pose_lhl_residues[i] = utils.convert_pdb_to_pose_resnums(pose, pdb_pose_lhl_residues[i], verbose=args.verbose)
        pose_helix_residues = []
        for i in range(len(pdb_pose_helix_residues)):
            pose_helix_residues.append([])
            pose_helix_residues[i] = utils.convert_pdb_to_pose_resnums(pose, pdb_pose_helix_residues[i], verbose=args.verbose)

        # Load strings containing residue ranges for PyRosetta
        pose_non_bb_remodeled_residues_string_list, pose_bb_remodeled_residues_string_list = \
            utils.load_pyrosetta_strings(design_path, pose_lhl_residues)
        
        if args.verbose:
            print(f'{design_path} bb_rem and non-bb_rem residues:')
            print(pose_non_bb_remodeled_residues_string_list)
            print(pose_bb_remodeled_residues_string_list)

        helix_idxs = [i for i in range(len(pose_helix_residues))]

        if args.verbose:
            print('Helix Indices:')
            print(helix_idxs)

        # If aligning on specified residues
        if args.use_align_residues_file:
            # Returns a list of two lists, containing 1) the reference structure's and 2) the design's residues on which to align.
            # Each sublist contains a list of contiguous residues on which to align, for utils.align_by_residues
            pdb_align_residues = get_align_residues(args, ref_path, design_path)

            if args.verbose:
                print('Residues on which to align:')
                print(f'{ref_path}: {pdb_align_residues[0]}')
                print(f'{design_path}: {pdb_align_residues[1]}\n')

            # Convert the align residues PDB-numbered residues to PyRosetta pose numbering
            print(f'Converting align_residues to PyRosetta Pose numbering for {ref_path} and {design_path}:')
            align_residues = [[],[]]
            for i in range(len(pdb_align_residues[0])):
                align_residues[0].append([])
                align_residues[0][i] = utils.convert_pdb_to_pose_resnums(ref_pose, pdb_align_residues[0][i], verbose=args.verbose)
            for i in range(len(pdb_align_residues[1])):
                align_residues[1].append([])
                align_residues[1][i] = utils.convert_pdb_to_pose_resnums(pose, pdb_align_residues[1][i], verbose=args.verbose)

            pdb_ref_all_sheet_residues = pdb_align_residues[0] # PDB-numbered sheet residues
            pdb_pose_all_sheet_residues = pdb_align_residues[1]
            ref_all_sheet_residues = align_residues[0] # 0-indexed sheet residues
            pose_all_sheet_residues = align_residues[1]

        # Otherwise, align on non-reshaped residues by default
        else:
            align_residues = None # 12.17.24: these should be set to non-bb_rem residues outside the RMSD function
            pdb_ref_all_sheet_residues = None
            pdb_pose_all_sheet_residues = None
            ref_all_sheet_residues = None
            pose_all_sheet_residues = None

        ref_align_residues = ref_all_sheet_residues
        pose_align_residues = pose_all_sheet_residues

        #####
        ### Calculate Helix RMSDs with my Helix RMSD functions
        ### This is necessary to find the trimmed helix residues for utils.steph_calculate_RMSDs_on_beta_only_helix(), too.
        #####

        # Align the design and reference structures on their align_residues, if defined
        if align_residues != None:
            # superimpose_poses_by_residues expects a single list of residues on which to align.
            pose, ref_pose = utils.superimpose_poses_by_residues(
<<<<<<< HEAD
                pose, 
                [i for lst in align_residues[1] for i in lst], 
                ref_pose, 
                [i for lst in align_residues[0] for i in lst], 
=======
                pose,
                [i for lst in align_residues[1] for i in lst],
                ref_pose,
                [i for lst in align_residues[0] for i in lst],
>>>>>>> be02a1e (lucs_af refactor and cleanup)
                atom_types=['N', 'CA', 'C'])

        # If align_residues is not defined, align structures on their non-remodeled residues.
        else:
            pose, ref_pose = utils.superimpose_poses_by_residues(
<<<<<<< HEAD
                pose, 
                pose_non_bb_remodeled_residues, 
                ref_pose, 
                ref_non_bb_remodeled_residues, 
                atom_types=['N', 'CA', 'C'])
    
        helix_rmsd_atom_types = ['N','CA','C']
        helix_rmsd, trimmed_ref_helix_residues, trimmed_pose_helix_residues = utils.calculate_bb_remodeled_region_rmsd(args,
=======
                pose,
                pose_non_bb_remodeled_residues,
                ref_pose,
                ref_non_bb_remodeled_residues,
                atom_types=['N', 'CA', 'C'])
    
        helix_rmsd_atom_types = ['N', 'CA', 'C']
        helix_rmsd, trimmed_ref_helix_residues, trimmed_pose_helix_residues = utils.calculate_bb_remodeled_region_rmsd(
            args,
>>>>>>> be02a1e (lucs_af refactor and cleanup)
            ref_path,
            design_path,
            helix_idxs=helix_idxs,
            is_af2_design=False,
            atom_types=helix_rmsd_atom_types,
<<<<<<< HEAD
            ref_lhl_residues=ref_lhl_residues, # 2.20.24: This may actually need to be a list of list of residues, considering that seg_i = ref_bb_remodeled_residues[helix_idx]
=======
            ref_lhl_residues=ref_lhl_residues,  # 2.20.24: This may actually need to be a list of list of residues, considering that seg_i = ref_bb_remodeled_residues[helix_idx]
>>>>>>> be02a1e (lucs_af refactor and cleanup)
            pose_lhl_residues=pose_lhl_residues,
            ref_helix_residues=ref_helix_residues,
            pose_helix_residues=pose_helix_residues,
            align_residues=align_residues)

        # Save the Ben Helix RMSD to the ret dict
        ret['ben_helix_rmsd'].append(helix_rmsd)

        if args.verbose:
            print(f"{ref_path} to {design_path} Helix RMSD (Ben's alignment and RMSD): {helix_rmsd}\n")

        # Calculate Helix RMSDs for each individual reshaped helix
        for helix_idx in helix_idxs:
<<<<<<< HEAD
            single_helix_rmsd, trimmed_pose1_single_helix_residues, trimmed_pose2_single_helix_residues = utils.calculate_bb_remodeled_region_rmsd(args,
                    ref_path,
                    design_path,
                    helix_idxs=[helix_idx],
                    is_af2_design=False,
                    atom_types=helix_rmsd_atom_types,
                    ref_lhl_residues=ref_lhl_residues,
                    pose_lhl_residues=pose_lhl_residues,
                    ref_helix_residues=ref_helix_residues,
                    pose_helix_residues=pose_helix_residues,
                    align_residues=align_residues)
=======
            single_helix_rmsd, trimmed_pose1_single_helix_residues, trimmed_pose2_single_helix_residues = utils.calculate_bb_remodeled_region_rmsd(
                args,
                ref_path,
                design_path,
                helix_idxs=[helix_idx],
                is_af2_design=False,
                atom_types=helix_rmsd_atom_types,
                ref_lhl_residues=ref_lhl_residues,
                pose_lhl_residues=pose_lhl_residues,
                ref_helix_residues=ref_helix_residues,
                pose_helix_residues=pose_helix_residues,
                align_residues=align_residues)
>>>>>>> be02a1e (lucs_af refactor and cleanup)

            if args.verbose:
                print(f"{ref_path} to {design_path} Helix {helix_idx} RMSD (Ben's alignment and RMSD): {single_helix_rmsd}")

            # Save the Ben Indiv Helix RMSDs to the ret dict
            if f'ben_indiv_helix{helix_idx}_rmsd' not in ret:
                ret[f'ben_indiv_helix{helix_idx}_rmsd'] = [single_helix_rmsd]
            else:
                ret[f'ben_indiv_helix{helix_idx}_rmsd'].append(single_helix_rmsd)
            
        #####
        ### Alignment 2 and Common Helix RMSD functions
        #####
        # 1. Align design and ss poses with utils.steph_align_by_residues
            # - Use lists of strings to select non-bb-rem or align residues
        # 2. Calculate Helix RMSDs on the aligned poses and their fd_helix_residues and ss_helix_residues with steph_calculate_RMSDs_on_beta_only_helix()
        # 3. Find the longest common helix indices with utils.steph_get_shared_helical_resn()
        # 4. Calculate the Helix RMSDs on the longest common helices with utils.steph_calculate_RMSDs_on_beta_only_helix()
        # Repeat this for AF2 Helix RMSDs (to design and ss) below.

        # If align_residues is not defined, align structures on their non-remodeled residues.
        if align_residues == None:
            ref_align_residues = utils.split_contiguous_residues(ref_non_bb_remodeled_residues)
            pose_align_residues = utils.split_contiguous_residues(pose_non_bb_remodeled_residues)

        ref_align_res_str_list = [f'{res[0]}-{res[-1]}' for res in ref_align_residues]
        pose_align_res_str_list = [f'{res[0]}-{res[-1]}' for res in pose_align_residues]

        if args.verbose:
            print('ref_align_res_str_list:')
            print(ref_align_res_str_list)
            print('pose_align_res_str_list:')
            print(pose_align_res_str_list)

        # Align the design and reference structures on their non-remodeled residues
        # Align pose2 to pose1
        pose, ref_pose = utils.steph_align_by_residues(
<<<<<<< HEAD
            pose, ref_pose, 
            pose_align_res_str_list, ref_align_res_str_list)

        # Define lists of strings of the helix residues for the design and reference structures
        # Ensure that each list of helix residues is of the same length, for utils.steph_calculate_RMSDs_on_beta_only_helix()			
        pose_helix_residues_str_list = [f'{res[0]}-{res[-1]}' for res in trimmed_pose_helix_residues]
        ref_helix_residues_str_list = [f'{res[0]}-{res[-1]}' for res in trimmed_ref_helix_residues]
        
=======
            pose, ref_pose,
            pose_align_res_str_list, ref_align_res_str_list)

        # Define lists of strings of the helix residues for the design and reference structures
        # Ensure that each list of helix residues is of the same length, for utils.steph_calculate_RMSDs_on_beta_only_helix()
        pose_helix_residues_str_list = [f'{res[0]}-{res[-1]}' for res in trimmed_pose_helix_residues]
        ref_helix_residues_str_list = [f'{res[0]}-{res[-1]}' for res in trimmed_ref_helix_residues]

>>>>>>> be02a1e (lucs_af refactor and cleanup)
        # Calculate Helix RMSDs between FastDesign and Starting Struct
        # This function uses rmsd_protein_bb_heavy_including_O
        # This function also requires that the residue selectors have the same number of residues
        helix_rmsds2 = utils.steph_calculate_RMSDs_on_beta_only_helix(
<<<<<<< HEAD
            pose, ref_pose, 
=======
            pose, ref_pose,
>>>>>>> be02a1e (lucs_af refactor and cleanup)
            pose_helix_residues_str_list, ref_helix_residues_str_list)

        # Save the Steph Helix RMSDs to the ret dict
        for h_idx, rmsd in enumerate(helix_rmsds2):
            if f'indiv_helix{h_idx}_rmsd2' not in ret:
                ret[f'indiv_helix{h_idx}_rmsd2'] = [rmsd]
            else:
                ret[f'indiv_helix{h_idx}_rmsd2'].append(rmsd)

            if args.verbose:
                print(f'Indiv Helix {h_idx} RMSD2 {ref_path} to {design_path}: {rmsd}')

        # Find the longest common helix indices between the design and reference structures

        ###
<<<<<<< HEAD
        # This function returns none if no common helix residues are found!
        ###
        
        pose_shared_helix_resn, ref_shared_helix_resn = utils.steph_get_shared_helical_resn(
            pose, ref_pose, 
=======
        # This function returns None if no common helix residues are found!
        ###

        pose_shared_helix_resn, ref_shared_helix_resn = utils.steph_get_shared_helical_resn(
            pose, ref_pose,
>>>>>>> be02a1e (lucs_af refactor and cleanup)
            pose_lhl_residues, ref_lhl_residues)

        # Define the shared helix residues as lists of strings
        pose_shared_helix_resn_str_list = [f'{res[0]}-{res[-1]}' if len(res) > 1 else None for res in pose_shared_helix_resn]
        ref_shared_helix_resn_str_list = [f'{res[0]}-{res[-1]}' if len(res) > 1 else None for res in ref_shared_helix_resn]

        # Calculate the Helix RMSDs on the longest common helices
        shared_helix_rmsds = utils.steph_calculate_RMSDs_on_beta_only_helix(
<<<<<<< HEAD
            pose, ref_pose, 
=======
            pose, ref_pose,
>>>>>>> be02a1e (lucs_af refactor and cleanup)
            pose_shared_helix_resn_str_list, ref_shared_helix_resn_str_list)

        # Save the Indiv Common Helix RMSDs to the ret dict
        for h_idx, rmsd in enumerate(shared_helix_rmsds):
            if f'indiv_common_helix{h_idx}_rmsd' not in ret:
                ret[f'indiv_common_helix{h_idx}_rmsd'] = [rmsd]
            else:
                ret[f'indiv_common_helix{h_idx}_rmsd'].append(rmsd)

            if args.verbose:
                print(f'Indiv Common Helix {h_idx} RMSD {ref_path} to {design_path}: {rmsd}')

        ###
        ### End Calculating RMSDs with Steph's Helix RMSD functions
        ###

        completed_rmsd_count += 1

        if args.test_stop != None and completed_rmsd_count >= args.test_stop:
            force_stop = True
            break

    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate all-by-all RMSDs for helix regions of designs.')

    # I/O arguments
    parser.add_argument('--pdb_dir', nargs='*', type=str, required=True,
<<<<<<< HEAD
        help="Path to a directory or directories containing subdir-split designs (subdirs named by design ID)\
            containing designs' pdb files.")
    parser.add_argument('--design_info_dir', nargs='*', type=str, required=True,
        help='Path to a directory or directories containing subdir-split designs (subdirs named by design ID)\
            containing design_info.json files.')
    parser.add_argument('--outfile', type=str, required=True, help='Path to a .csv file to save results.')
    parser.add_argument('--design_id_suffix', type=str, required=False, default='',
        help='Suffix added to <design_id>/<design_id><suffix>.pdb file names.')
    parser.add_argument('--skip_completed_files', action='store_true', 
                    help='If the output files for the current task already exist, then skip the calculations.')
    parser.add_argument('--include_designs_df', type=str, required=False, default=None,
			help='Path to a .csv file with a design_id column. If provided, only these design IDs '
			'will be analyzed.')
    
    # AlphaFold inputs
    parser.add_argument('--analyze_af2', action='store_true',
			help="Instead of using the PDB files in the --pdb_dir, find these designs'\
                associated AF2 predictions, provided in the --af2_data_dirs argument.")
    parser.add_argument('--af2_data_parents', nargs='*', type=str, required=False, default=None,
			help='Path to a parent directory or parent directories containing subdirectories with AF2 \
            predictions for LUCS designs. Only used if --analyze_af2 argument is passed.')
    parser.add_argument('--is_colabfold_format', action='store_true',
			help='Indicate whether the AF2 data dir is in Colabfold output format (all PDB and json\
			files for all designs in the same directory).')
    parser.add_argument('--af2_suffixes', nargs='*', type=str, required=False, default=[''],
			help='A list of suffixes used to name AF2 predictions, such as DESIGNID_SUFFIX, where\
                 DESIGNIDs are the subdir names in the --pdb_dir directories. Ex. DEISGNID_mpnnSeq5')
    parser.add_argument('--split_af2_fnames_at', type=str, required=False, default='_unrelaxed_',
            help='String by which to split AF2 prediction files to get design_ids. Default is _unrelaxed_.\
                Passing an empty string will not split the filenames.')
    
    # AlphaFold3 inputs
    parser.add_argument('--is_af3', action='store_true',
            help='Indicate whether the AlphaFold predictions are from AlphaFold3.')
    parser.add_argument('--af_prefix', type=str, required=False, default='',
            help='Prefix in front of design_id in AF2/3 predictions, such as fold_ for AF3 server outputs.')
    parser.add_argument('--af_is_cif', action='store_true',
            help='Indicate that the AF predictions are .cif files.')
=======
        help="Path to a directory or directories containing subdir-split designs (subdirs named by design ID) "
             "containing designs' PDB files.")
    parser.add_argument('--design_info_dir', nargs='*', type=str, required=True,
        help='Path to a directory or directories containing subdir-split designs (subdirs named by design ID) '
             'containing design_info.json files.')
    parser.add_argument('--outfile', type=str, required=True, help='Path to a .csv file to save results.')
    parser.add_argument('--design_id_suffix', type=str, required=False, default='',
        help='Suffix added to <design_id>/<design_id><suffix>.pdb file names.')
    parser.add_argument('--skip_completed_files', action='store_true',
        help='If the output files for the current task already exist, then skip the calculations.')
    parser.add_argument('--include_designs_df', type=str, required=False, default=None,
        help='Path to a .csv file with a design_id column. If provided, only these design IDs '
             'will be analyzed.')
    
    # AlphaFold inputs
    parser.add_argument('--analyze_af2', action='store_true',
        help="Instead of using the PDB files in the --pdb_dir, find these designs' "
             "associated AF2 predictions, provided in the --af2_data_dirs argument.")
    parser.add_argument('--af2_data_parents', nargs='*', type=str, required=False, default=None,
        help='Path to a parent directory or parent directories containing subdirectories with AF2 '
             'predictions for LUCS designs. Only used if --analyze_af2 argument is passed.')
    parser.add_argument('--is_colabfold_format', action='store_true',
        help='Indicate whether the AF2 data dir is in ColabFold output format (all PDB and json '
             'files for all designs in the same directory).')
    parser.add_argument('--af2_suffixes', nargs='*', type=str, required=False, default=[''],
        help='A list of suffixes used to name AF2 predictions, such as DESIGNID_SUFFIX, where '
             'DESIGNIDs are the subdir names in the --pdb_dir directories. Ex. DESIGNID_mpnnSeq5')
    parser.add_argument('--split_af2_fnames_at', type=str, required=False, default='_unrelaxed_',
        help='String by which to split AF2 prediction files to get design_ids. Default is _unrelaxed_. '
             'Passing an empty string will not split the filenames.')
    
    # AlphaFold3 inputs
    parser.add_argument('--is_af3', action='store_true',
        help='Indicate whether the AlphaFold predictions are from AlphaFold3.')
    parser.add_argument('--af_prefix', type=str, required=False, default='',
        help='Prefix in front of design_id in AF2/3 predictions, such as fold_ for AF3 server outputs.')
    parser.add_argument('--af_is_cif', action='store_true',
        help='Indicate that the AF predictions are .cif files.')
>>>>>>> be02a1e (lucs_af refactor and cleanup)

    # Alignment and RMSD arguments
    parser.add_argument('--reference_structure_path', type=str, required=False,
        default='/wynton/home/kortemme/borr/thesis/local_protein_sequence_design/test_inputs/2lv8_inputs/2lv8_cleaned.pdb',
        help='Path to the reference structure.')
    parser.add_argument('--reference_structure_design_info', type=str, required=False,
        default='/wynton/home/kortemme/borr/thesis/local_protein_sequence_design/test_inputs/2lv8_inputs/2lv8_insertion_points.json',
        help="Path to the reference structure's insertion points file.")
    parser.add_argument('--reference_structure_res_start_idx', type=int, required=False, default=1,
        help="Index on which the reference structure's residue numbers begin.")
<<<<<<< HEAD
    parser.add_argument('--use_dssp', action='store_true', help='Use DSSP to find helical residues.')
    parser.add_argument('--assume_helix', action='store_true', help='Assume all residues are helical.')
    parser.add_argument('--bb_rem_all_helical', action='store_true', help='Consider all helical residues as bb-remodeled.')
    parser.add_argument('--use_align_residues_file', action='store_true', help='Use align_residues.json file for structure alignment.')
    parser.add_argument('--aligned_pdb_outdir', type=str, required=False, default=None,
        help='If provided, then save aligned PDB files when calculating Helix RMSD to the reference structure \
        to the specified directory.')
    parser.add_argument('--reference_structure_helix_residues_file', type=str, required=False, default=None,
			help='Use a provided json file containing (a list of lists of) helix residues in the Starting Structure. \
			If not provided, then use DSSP to find reshaped helix residues in the Starting Structure.')
=======
    parser.add_argument('--use_dssp', action='store_true',
        help='Use DSSP to find helical residues.')
    parser.add_argument('--assume_helix', action='store_true',
        help='Assume all residues are helical.')
    parser.add_argument('--bb_rem_all_helical', action='store_true',
        help='Consider all helical residues as backbone-remodeled.')
    parser.add_argument('--use_align_residues_file', action='store_true',
        help='Use align_residues.json file for structure alignment.')
    parser.add_argument('--aligned_pdb_outdir', type=str, required=False, default=None,
        help='If provided, then save aligned PDB files when calculating Helix RMSD to the reference structure '
             'to the specified directory.')
    parser.add_argument('--reference_structure_helix_residues_file', type=str, required=False, default=None,
        help='Use a provided JSON file containing (a list of lists of) helix residues in the Starting Structure. '
             'If not provided, then use DSSP to find reshaped helix residues in the Starting Structure.')
>>>>>>> be02a1e (lucs_af refactor and cleanup)

    # Parallelization arguments
    parser.add_argument('--num_tasks', type=int, required=False, default=1,
        help='Number of tasks by which to divide the calculations.')
    parser.add_argument('--task_id', type=int, required=False, default=1,
        help='ID of the current task (1-indexed, e.g. $SGE_TASK_ID for SGE job distributor).')

    # Debugging arguments
<<<<<<< HEAD
    parser.add_argument('--verbose', action='store_true', help='Print verbose output.')
=======
    parser.add_argument('--verbose', action='store_true',
        help='Print verbose output.')
>>>>>>> be02a1e (lucs_af refactor and cleanup)
    parser.add_argument('--test_stop', type=int, required=False, default=None,
        help='Number of test calculations to perform.')

    args = parser.parse_args()

    # If the output file for the current task already exists, then skip the calculations
    if args.skip_completed_files:
        files_exist = check_completed_files(args)
        if files_exist:
            print(f"Output files for task {args.task_id} already exist. Skipping calculations.")
            exit()

    if args.analyze_af2:
        design_paths_ids = get_af2_design_paths(args)
    else:
        design_paths_ids = get_design_paths(args)

    if args.verbose:
        print(f'Number of Design Paths: {len(design_paths_ids)}')
        print(f'Design Paths[:10]: {design_paths_ids[:10]}')

    ret = calculate_helix_rmsds_to_ref(args, design_paths_ids)

    # Print the lengths of the arrays in the results dict
    if args.verbose:
        print(f"Results lengths:")
        for key in ret:
            print(f"{key}: {len(ret[key])}")

    # Convert the results dict to a pandas df
    df = pd.DataFrame(ret)

    # Save results to files
    if args.num_tasks != 1:
        fname_append = f'_{args.task_id}'
    else:
        fname_append = ''

    # Save a dataframe of the design paths
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    df_outpath = os.path.join(args.outfile.replace('.csv', f'{fname_append}.csv'))
    df.to_csv(df_outpath, index=False)
    print(f"Results saved to {df_outpath}")