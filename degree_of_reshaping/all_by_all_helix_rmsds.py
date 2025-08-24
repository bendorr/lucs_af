"""
all_by_all_helix_rmsds.py

Ben Orr
2.6.25

Calculate all-by-all helix RMSDs between specified
helices in provided PDB files.

"""

import os
import json
import argparse
import utils_degree_reshaped as utils
import numpy as np
import pandas as pd

from pyrosetta import init, pose_from_file
# init pyrosetta
init(options='-mute all')

#####

def check_completed_files(args):
    if args.num_tasks != 1:
        fname_append = f'_task{args.task_id}_of_{args.num_tasks}'
    else:
        fname_append = ''
    if os.path.exists(os.path.join(args.output_dir, f'design_paths{fname_append}.json')) and \
        os.path.exists(os.path.join(args.output_dir, f'helix_rmsd_results{fname_append}.npy')) and \
            os.path.exists(os.path.join(args.output_dir, f'indiv_helix_rmsd_results{fname_append}.npy')) and \
                os.path.exists(os.path.join(args.output_dir, f'indiv_helix_rmsd2_results{fname_append}.npy')) and \
                    os.path.exists(os.path.join(args.output_dir, f'indiv_common_helix_rmsd_results{fname_append}.npy')):
        return True
    return False

def get_design_paths(args):
    design_paths = []
    if args.include_designs_df != None:
        include_designs_df = pd.read_csv(args.include_designs_df)
        include_design_ids = include_designs_df['design_id'].tolist()

    for data_dir in args.data_dir:
        for subdir in os.listdir(data_dir):
            if args.include_designs_df != None:
                if subdir not in include_design_ids:
                    continue
            design_path = os.path.join(data_dir, subdir, f"{subdir}{args.design_id_suffix}.pdb")
            if os.path.exists(design_path):
                design_paths.append(design_path)

    # Sort the design paths
    design_paths.sort()

    return design_paths

def get_design_info_path(design_path):
    fnames = [design_path.replace('.pdb', '_design_info.json'),
              os.path.join(design_path.rsplit('/', 1)[0],'design_info.json')]
    for fname in fnames:
        if os.path.exists(fname):
            return fname
    raise FileNotFoundError(f"Design info file not found for {design_path}")

def get_align_residues(args, design_path1, design_path2):
    # Look in the design's folder for a {design_id}_align_residues.json file
    design1_align_res_file = design_path1.replace('.pdb','_align_residues.json') 

    # Look in the design's folder for a {design_id}_align_residues.json file
    design2_align_res_file = design_path2.replace('.pdb','_align_residues.json') 

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
    align_residues = [[],[]]
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

def calculate_all_by_all_helix_rmsds(args, design_paths):

    # Initialize results matrices
    helix_rmsd_results = np.zeros((len(design_paths), len(design_paths)))
    indiv_helix_rmsd_results = [np.zeros((len(design_paths), len(design_paths))), 
                              np.zeros((len(design_paths), len(design_paths)))]

    indiv_helix_rmsd2_results = [np.zeros((len(design_paths), len(design_paths))),
                                 np.zeros((len(design_paths), len(design_paths)))]

    indiv_common_helix_rmsd_results = [np.zeros((len(design_paths), len(design_paths))),
                                        np.zeros((len(design_paths), len(design_paths)))]

    task_idx = 0
    completed_rmsd_count = 0
    force_stop = False
    for design_i, design_path1 in enumerate(design_paths):
        if force_stop:
            break
        # Load design1
        pose1 = pose_from_file(design_path1)

        # Get design_info_path1
        design_info_path1 = get_design_info_path(design_path1)

        # Get PDB-numbered non-backbone remodeled residues
        pdb_pose1_bb_remodeled_residues, pdb_pose1_non_bb_remodeled_residues = \
			utils.get_remodeled_residues_from_design_info(
				pdb_file = design_path1,
				design_info_file = design_info_path1)
        
        if args.verbose:
            print(f'{design_path1} bb_remodeled_residues (PDB-numbered):')
            print(f"{pdb_pose1_bb_remodeled_residues}")
            print(f'{design_path1} non_bb_remodeled_residues (PDB-numbered):')
            print(f"{pdb_pose1_non_bb_remodeled_residues}\n")

        # Find the helix residues of design1. This list has len(num_lhls)
        pdb_pose1_lhl_residues, pdb_pose1_helix_residues = utils.get_lhl_residues(
                design_info_file = design_info_path1,
                seq_len = len(pose1.sequence()),
                pose=pose1,
                use_dssp=args.use_dssp,
                assume_helix=args.assume_helix,
                bb_rem_all_helical=args.bb_rem_all_helical)
        
        if args.verbose:
            print(f'{design_path1} LHL residues:')
            for res_list in pdb_pose1_lhl_residues:
                print(res_list)
            print(f'{design_path1} Helix residues:')
            for res_list in pdb_pose1_helix_residues:
                print(res_list)

        # Convert between PDB and PyRosetta residue numberings for design1
        print(f'Converting PDB to Rosetta numbering for {design_path1}:')
        pose1_bb_remodeled_residues = utils.convert_pdb_to_pose_resnums(pose1, pdb_pose1_bb_remodeled_residues, verbose=args.verbose)
        pose1_non_bb_remodeled_residues = utils.convert_pdb_to_pose_resnums(pose1, pdb_pose1_non_bb_remodeled_residues, verbose=args.verbose)
        pose1_lhl_residues = []
        for i in range(len(pdb_pose1_lhl_residues)):
            pose1_lhl_residues.append([])
            pose1_lhl_residues[i] = utils.convert_pdb_to_pose_resnums(pose1, pdb_pose1_lhl_residues[i], verbose=args.verbose)
        pose1_helix_residues = []
        for i in range(len(pdb_pose1_helix_residues)):
            pose1_helix_residues.append([])
            pose1_helix_residues[i] = utils.convert_pdb_to_pose_resnums(pose1, pdb_pose1_helix_residues[i], verbose=args.verbose)

        # Load strings containing residue ranges for PyRosetta
        pose1_non_bb_remodeled_residues_string_list, pose1_bb_remodeled_residues_string_list = \
            utils.load_pyrosetta_strings(design_path1, pose1_lhl_residues)
        
        if args.verbose:
            print(f'{design_path1} bb_rem and non-bb_rem residues:')
            print(pose1_non_bb_remodeled_residues_string_list)
            print(pose1_bb_remodeled_residues_string_list)
               
        for design_j, design_path2 in enumerate(design_paths):
            
            if design_i < design_j:

                task_idx += 1
                if task_idx % args.num_tasks != (args.task_id-1):
                    continue

                print(f"\n\nCalculating RMSD index {task_idx}\nTask {args.task_id} of {args.num_tasks}\nDesign ({design_i}, {design_j}) of ({len(design_paths)}, {len(design_paths)})\n{design_path1} to {design_path2}\n\n")

                # Load design2
                pose2 = pose_from_file(design_path2)
                # Get design_info_path2
                design_info_path2 = get_design_info_path(design_path2)

                # Get PDB-numbered non-backbone remodeled residues
                pdb_pose2_bb_remodeled_residues, pdb_pose2_non_bb_remodeled_residues = \
                    utils.get_remodeled_residues_from_design_info(
                        pdb_file = design_path2,
                        design_info_file = design_info_path2)
                
                if args.verbose:
                    print(f'{design_path2} bb_remodeled_residues (PDB-numbered):')
                    print(f"{pdb_pose2_bb_remodeled_residues}")
                    print(f'{design_path2} non_bb_remodeled_residues (PDB-numbered):')
                    print(f"{pdb_pose2_non_bb_remodeled_residues}\n")

                # Find the helix residues of design2. This list has len(num_lhls)
                pdb_pose2_lhl_residues, pdb_pose2_helix_residues = utils.get_lhl_residues(
                        design_info_file = design_info_path2,
                        seq_len = len(pose2.sequence()),
                        pose=pose2,
                        use_dssp=args.use_dssp,
                        assume_helix=args.assume_helix,
                        bb_rem_all_helical=args.bb_rem_all_helical)

                if args.verbose:
                    print(f'{design_path2} LHL residues:')
                    for res_list in pdb_pose2_lhl_residues:
                        print(res_list)
                    print(f'{design_path2} Helix residues:')
                    for res_list in pdb_pose2_helix_residues:
                        print(res_list)
                
                # Convert between PDB and PyRosetta residue numberings for design2
                pose2_bb_remodeled_residues = utils.convert_pdb_to_pose_resnums(pose2, pdb_pose2_bb_remodeled_residues, verbose=args.verbose)
                pose2_non_bb_remodeled_residues = utils.convert_pdb_to_pose_resnums(pose2, pdb_pose2_non_bb_remodeled_residues, verbose=args.verbose)
                pose2_lhl_residues = []
                for i in range(len(pdb_pose2_lhl_residues)):
                    pose2_lhl_residues.append([])
                    pose2_lhl_residues[i] = utils.convert_pdb_to_pose_resnums(pose2, pdb_pose2_lhl_residues[i], verbose=args.verbose)
                pose2_helix_residues = []
                for i in range(len(pdb_pose2_helix_residues)):
                    pose2_helix_residues.append([])
                    pose2_helix_residues[i] = utils.convert_pdb_to_pose_resnums(pose2, pdb_pose2_helix_residues[i], verbose=args.verbose)
                # Load strings containing residue ranges for PyRosetta
                pose2_non_bb_remodeled_residues_string_list, pose2_bb_remodeled_residues_string_list = \
                    utils.load_pyrosetta_strings(design_path2, pose2_lhl_residues)
                if args.verbose:
                    print(f'{design_path2} bb_rem and non-bb_rem residues:')
                    print(pose2_non_bb_remodeled_residues_string_list)
                    print(pose2_bb_remodeled_residues_string_list)

                helix_idxs = [i for i in range(len(pose1_helix_residues))]

                if args.verbose:
                    print('Helix Indices:')
                    print(helix_idxs)

                # If aligning on specified residues
                if args.use_align_residues_file:
                    # Returns a list of two lists, containing 1) the reference structure's and 2) the design's residues on which to align.
                    # Each sublist contains a list of contiguous residues on which to align, for utils.align_by_residues
                    pdb_align_residues = get_align_residues(args, design_path1, design_path2)

                    if args.verbose:
                        print('Residues on which to align:')
                        print(f'{design_path1}: {pdb_align_residues[0]}')
                        print(f'{design_path2}: {pdb_align_residues[1]}\n')

                    # Convert the align residues PDB-numbered residues to PyRosetta pose numbering
                    print(f'Converting align_residues to PyRosetta Pose numbering for {design_path1} and {design_path2}:')
                    align_residues = [[],[]]
                    for i in range(len(pdb_align_residues[0])):
                        align_residues[0].append([])
                        align_residues[0][i] = utils.convert_pdb_to_pose_resnums(pose1, pdb_align_residues[0][i], verbose=args.verbose)
                    for i in range(len(pdb_align_residues[1])):
                        align_residues[1].append([])
                        align_residues[1][i] = utils.convert_pdb_to_pose_resnums(pose2, pdb_align_residues[1][i], verbose=args.verbose)

                    pdb_pose1_all_sheet_residues = pdb_align_residues[0] # PDB-numbered sheet residues
                    pdb_pose2_all_sheet_residues = pdb_align_residues[1]
                    pose1_all_sheet_residues = align_residues[0] # 0-indexed sheet residues
                    pose2_all_sheet_residues = align_residues[1]

                    pose1_align_residues = pose1_all_sheet_residues
                    pose2_align_residues = pose2_all_sheet_residues

                # Otherwise, align on non-reshaped residues by default
                else:
                    align_residues = None # these are set to non-bb_rem residues below
                    pdb_pose1_all_sheet_residues = None
                    pdb_pose2_all_sheet_residues = None
                    pose1_all_sheet_residues = None
                    pose2_all_sheet_residues = None

                #####
                ### Calculate Helix RMSDs with my Helix RMSD functions
                ### This is necessary to find the trimmed helix residues for utils.steph_calculate_RMSDs_on_beta_only_helix(), too.
                #####

                # Align the design and reference structures on their align_residues, if defined
                if align_residues != None:
                    # superimpose_poses_by_residues expects a single list of residues on which to align.
                    pose2, pose1 = utils.superimpose_poses_by_residues(
                        pose2, 
                        [i for lst in align_residues[1] for i in lst], 
                        pose1, 
                        [i for lst in align_residues[0] for i in lst], 
                        atom_types=['N', 'CA', 'C'])

                # If align_residues is not defined, align structures on their non-remodeled residues.
                else:
                    pose2, pose1 = utils.superimpose_poses_by_residues(
                        pose2, 
                        pose2_non_bb_remodeled_residues, 
                        pose1, 
                        pose1_non_bb_remodeled_residues, 
                        atom_types=['N', 'CA', 'C'])
            
                helix_rmsd_atom_types = ['N','CA','C']
                helix_rmsd, trimmed_pose1_helix_residues, trimmed_pose2_helix_residues = utils.calculate_bb_remodeled_region_rmsd(args,
					design_path1,
					design_path2,
					helix_idxs=helix_idxs,
					is_af2_design=False,
					atom_types=helix_rmsd_atom_types,
					ref_lhl_residues=pose1_lhl_residues,
					pose_lhl_residues=pose2_lhl_residues,
					ref_helix_residues=pose1_helix_residues,
					pose_helix_residues=pose2_helix_residues,
					align_residues=align_residues)
                
                ben_results[design_i, design_j] = helix_rmsd
                ben_results[design_j, design_i] = helix_rmsd  # RMSD is symmetric

                if args.verbose:
                    print(f"{design_path1} to {design_path2} Helix RMSD (Ben's alignment and RMSD): {helix_rmsd}\n")

                # Calculate Helix RMSDs for each individual reshaped helix
                for helix_idx in helix_idxs:
                    single_helix_rmsd, trimmed_pose1_single_helix_residues, trimmed_pose2_single_helix_residues = utils.calculate_bb_remodeled_region_rmsd(args,
                            design_path1,
                            design_path2,
                            helix_idxs=[helix_idx],
                            is_af2_design=False,
                            atom_types=helix_rmsd_atom_types,
                            ref_lhl_residues=pose1_lhl_residues,
                            pose_lhl_residues=pose2_lhl_residues,
                            ref_helix_residues=pose1_helix_residues,
                            pose_helix_residues=pose2_helix_residues,
                            align_residues=align_residues)

                    if args.verbose:
                        print(f"{design_path1} to {design_path2} Helix {helix_idx} RMSD (Ben's alignment and RMSD): {single_helix_rmsd}")

                    indiv_helix_rmsd_results[helix_idx][design_i, design_j] = single_helix_rmsd
                    indiv_helix_rmsd_results[helix_idx][design_j, design_i] = single_helix_rmsd  # RMSD is symmetric

                #####
                ### Steph's Helix RMSD functions
                #####
                # 1. Align design and ss poses with utils.steph_align_by_residues
                    # - Use lists of strings to select non-bb-rem or align residues 
                # 2. Calculate Helix RMSDs on the aligned poses and their fd_helix_residues and ss_helix_residues with steph_calculate_RMSDs_on_beta_only_helix()
                # 3. Find the longest common helix indices with utils.steph_get_shared_helical_resn()
                # 4. Calculate the Helix RMSDs on the longest common helices with utils.steph_calculate_RMSDs_on_beta_only_helix()
                # Repeat this for AF Helix RMSDs (to design and ss) below.

                # If align_residues is not defined, align structures on their non-remodeled residues.
                if align_residues == None:
                    pose1_align_residues = utils.split_contiguous_residues(pose1_non_bb_remodeled_residues)
                    pose2_align_residues = utils.split_contiguous_residues(pose2_non_bb_remodeled_residues)

                pose1_align_res_str_list = [f'{res[0]}-{res[-1]}' for res in pose1_align_residues]
                pose2_align_res_str_list = [f'{res[0]}-{res[-1]}' for res in pose2_align_residues]

                if args.verbose:
                    print('pose1_align_res_str_list:')
                    print(pose1_align_res_str_list)
                    print('pose2_align_res_str_list:')
                    print(pose2_align_res_str_list)

                # Align the design and reference structures on their non-remodeled residues
                # Align pose2 to pose1
                pose2, pose1 = utils.steph_align_by_residues(
                    pose2, pose1, 
                    pose2_align_res_str_list, pose1_align_res_str_list)

                # Define lists of strings of the helix residues for the design and reference structures
                # Ensure that each list of helix residues is of the same length, for utils.steph_calculate_RMSDs_on_beta_only_helix()			
                pose2_helix_residues_str_list = [f'{res[0]}-{res[-1]}' for res in trimmed_pose2_helix_residues]
                pose1_helix_residues_str_list = [f'{res[0]}-{res[-1]}' for res in trimmed_pose1_helix_residues]
                
                # Calculate Helix RMSDs between FastDesign and Starting Struct
                # This function uses rmsd_protein_bb_heavy_including_O
                # This function also requires that the residue selectors have the same number of residues
                indiv_helix_rmsds2 = utils.steph_calculate_RMSDs_on_beta_only_helix(
                    pose2, pose1, 
                    pose2_helix_residues_str_list, pose1_helix_residues_str_list)

                # Save the Helix RMSDs to the sub_dict
                for h_idx, rmsd in enumerate(indiv_helix_rmsds2):
                    indiv_helix_rmsd_results[h_idx][design_i, design_j] = rmsd
                    indiv_helix_rmsd_results[h_idx][design_j, design_i] = rmsd # RMSD is symmetric
                    if args.verbose:
                        print(f'Indiv. Helix {h_idx} RMSD2 {design_path1} to {design_path2}: {rmsd}')

                # Find the longest common helix indices between the design and reference structures

                ###
                # This function returns none if no common helix residues are found!
                ###
                
                pose2_shared_helix_resn, pose1_shared_helix_resn = utils.steph_get_shared_helical_resn(
                    pose2, pose1, 
                    pose2_lhl_residues, pose1_lhl_residues)

                # Define the shared helix residues as lists of strings
                pose2_shared_helix_resn_str_list = [f'{res[0]}-{res[-1]}' if len(res) > 1 else None for res in pose2_shared_helix_resn]
                pose1_shared_helix_resn_str_list = [f'{res[0]}-{res[-1]}' if len(res) > 1 else None for res in pose1_shared_helix_resn]

                # Calculate the Helix RMSDs on the longest common helices
                shared_helix_rmsds = utils.steph_calculate_RMSDs_on_beta_only_helix(
                    pose2, pose1, 
                    pose2_shared_helix_resn_str_list, pose1_shared_helix_resn_str_list)

                # Save the shared Helix RMSDs to the sub_dict
                for h_idx, rmsd in enumerate(shared_helix_rmsds):
                    indiv_common_helix_rmsd_results[h_idx][design_i, design_j] = rmsd
                    indiv_common_helix_rmsd_results[h_idx][design_j, design_i] = rmsd # RMSD is symmetric
                    if args.verbose:
                        print(f'Indiv. Common Helix {h_idx} RMSD {design_path1} to {design_path2}: {rmsd}')

                ###
                ### End Calculating RMSDs with Steph's Helix RMSD functions
                ###

                completed_rmsd_count += 1

                if args.test_stop != None and completed_rmsd_count >= args.test_stop:
                    force_stop = True
                    break

    return helix_rmsd_results, indiv_helix_rmsd_results, indiv_helix_rmsd2_results, indiv_common_helix_rmsd_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate all-by-all RMSDs for helix regions of designs.')
    
    # I/O arguments
    parser.add_argument('--data_dir', nargs='*', type=str, required=True,
			help='Path to a directory or directories containing subdir-split designs (subdirs named by design ID).')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to a directory to save the results.')
    parser.add_argument('--include_designs_df', type=str, required=True, 
                        help="Path to a dataframe a 'design_id' column, with designs to include in the calculations.")
    parser.add_argument('--design_id_suffix', type=str, required=False, default='',
			help='Suffix added to <design_id>/<design_id><suffix>.pdb file names.')
    parser.add_argument('--skip_completed_files', action='store_true', 
                        help='If the output files for the current task already exist, then skip the calculations.')

    # Alignment and RMSD arguments
    parser.add_argument('--use_dssp', action='store_true', help='Use DSSP to find helical residues.')
    parser.add_argument('--assume_helix', action='store_true', help='If fewer than 4 helical residues are found, '
        'assume all LHL residues aside from the 5 N- and C-terminal residues are helical.')
    parser.add_argument('--bb_rem_all_helical', action='store_true', help='Consider all bb-remodeled residues to be helical.')
    parser.add_argument('--use_align_residues_file', action='store_true', help='Use align_residues.json file for structure alignment.')
    parser.add_argument('--aligned_pdb_outdir', type=str, required=False, default=None,
            help='If provided, then save aligned PDB files when calculating Helix RMSD to the reference structure \
            to the specified directory.')
    
    # Parallelization arguments
    parser.add_argument('--num_tasks', type=int, required=False, default=1,
            help='Number of tasks by which to divide the calculations.')
    parser.add_argument('--task_id', type=int, required=False, default=1,
            help='ID of the current task (1-indexed, e.g. $SGE_TASK_ID for SGE job distributor).')
    
    # Debugging arguments
    parser.add_argument('--verbose', action='store_true', help='Print verbose output.')
    parser.add_argument('--test_stop', type=int, required=False, default=None,
            help='Number of test calculations to perform.')
    
    args = parser.parse_args()

    # If the output file for the current task already exists, then skip the calculations
    if args.skip_completed_files:
        files_exist = check_completed_files(args)
        if files_exist:
            print(f"Output files for task {args.task_id} already exist. Skipping calculations.")
            exit()

    design_paths = get_design_paths(args)

    if args.verbose:
        print(f'Number of Design Paths: {len(design_paths)}')
        print(f'Design Paths[:10]: {design_paths[:10]}')

    helix_rmsd_results, indiv_helix_rmsd_results, indiv_helix_rmsd2_results, indiv_common_helix_rmsd_results = \
        calculate_all_by_all_helix_rmsds(args, design_paths)
    
    # Save results to files
    if args.num_tasks != 1:
        fname_append = f'_task{args.task_id}_of_{args.num_tasks}'
    else:
        fname_append = ''

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, f'design_paths{fname_append}.json'), 'w') as f:
        json.dump(design_paths, f)
    np.save(os.path.join(args.output_dir, f'helix_rmsd_results{fname_append}.npy'), helix_rmsd_results)
    np.save(os.path.join(args.output_dir, f'indiv_helix_rmsd_results{fname_append}.npy'), indiv_helix_rmsd_results)
    np.save(os.path.join(args.output_dir, f'indiv_helix_rmsd2_results{fname_append}.npy'), indiv_helix_rmsd2_results)
    np.save(os.path.join(args.output_dir, f'indiv_common_helix_rmsd_results{fname_append}.npy'), indiv_common_helix_rmsd_results)
    print(f"RMSD calculations completed and results saved to {args.output_dir}")

