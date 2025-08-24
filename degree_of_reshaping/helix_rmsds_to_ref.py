import os
import json
import argparse
import utils_degree_reshaped as utils
import numpy as np
import pandas as pd
from natsort import natsorted

from pyrosetta import init, pose_from_file
# init pyrosetta
init(options='-mute all')

#####

def check_completed_files(args):
    if args.num_tasks != 1:
        fname_append = f'_task{args.task_id}_of_{args.num_tasks}'
    else:
        fname_append = ''
    df_outpath = os.path.join(args.outfile.replace('.csv', f'{fname_append}.csv'))
    if os.path.exists(df_outpath):
        return True
    return False

def get_design_paths(args):
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

def get_af2_design_paths(args):
    # Return a list of tuples, containing design IDs and their AF2 subdirectories.

    # Each af_file corresponds to an RO2 sequence, unless af_dir is in colabfold
    # format, in which case each af_file is a PDB, json, or png file.
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
                    
    return design_paths_ids

def get_design_info_path(args, design_id):
    # args.design_info_dir is a list of dirs. Find the design's
    # subdir in one of these dirs.
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

def calculate_helix_rmsds_to_ref(args, design_paths_ids):
    """
    Calculate the helix RMSDs between the designs and the reference structure.
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
			utils.get_remodeled_residues_from_design_info(
				pdb_file = design_path,
				design_info_file = design_info_path)
        
        if args.verbose:
            print(f'{design_path} bb_remodeled_residues (PDB-numbered):')
            print(f"{pdb_pose_bb_remodeled_residues}")
            print(f'{design_path} non_bb_remodeled_residues (PDB-numbered):')
            print(f"{pdb_pose_non_bb_remodeled_residues}\n")

        # Find the helix residues of design1. This list has len(num_lhls)
        pdb_pose_lhl_residues, pdb_pose_helix_residues = utils.get_lhl_residues(
                design_info_file = design_info_path,
                seq_len = len(pose.sequence()),
                pose=pose,
                use_dssp=args.use_dssp,
                assume_helix=args.assume_helix,
                bb_rem_all_helical=args.bb_rem_all_helical)
        
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
                pose, 
                [i for lst in align_residues[1] for i in lst], 
                ref_pose, 
                [i for lst in align_residues[0] for i in lst], 
                atom_types=['N', 'CA', 'C'])

        # If align_residues is not defined, align structures on their non-remodeled residues.
        else:
            pose, ref_pose = utils.superimpose_poses_by_residues(
                pose, 
                pose_non_bb_remodeled_residues, 
                ref_pose, 
                ref_non_bb_remodeled_residues, 
                atom_types=['N', 'CA', 'C'])
    
        helix_rmsd_atom_types = ['N','CA','C']
        helix_rmsd, trimmed_ref_helix_residues, trimmed_pose_helix_residues = utils.calculate_bb_remodeled_region_rmsd(args,
            ref_path,
            design_path,
            helix_idxs=helix_idxs,
            is_af2_design=False,
            atom_types=helix_rmsd_atom_types,
            ref_lhl_residues=ref_lhl_residues, # 2.20.24: This may actually need to be a list of list of residues, considering that seg_i = ref_bb_remodeled_residues[helix_idx]
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
            pose, ref_pose, 
            pose_align_res_str_list, ref_align_res_str_list)

        # Define lists of strings of the helix residues for the design and reference structures
        # Ensure that each list of helix residues is of the same length, for utils.steph_calculate_RMSDs_on_beta_only_helix()			
        pose_helix_residues_str_list = [f'{res[0]}-{res[-1]}' for res in trimmed_pose_helix_residues]
        ref_helix_residues_str_list = [f'{res[0]}-{res[-1]}' for res in trimmed_ref_helix_residues]
        
        # Calculate Helix RMSDs between FastDesign and Starting Struct
        # This function uses rmsd_protein_bb_heavy_including_O
        # This function also requires that the residue selectors have the same number of residues
        helix_rmsds2 = utils.steph_calculate_RMSDs_on_beta_only_helix(
            pose, ref_pose, 
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
        # This function returns none if no common helix residues are found!
        ###
        
        pose_shared_helix_resn, ref_shared_helix_resn = utils.steph_get_shared_helical_resn(
            pose, ref_pose, 
            pose_lhl_residues, ref_lhl_residues)

        # Define the shared helix residues as lists of strings
        pose_shared_helix_resn_str_list = [f'{res[0]}-{res[-1]}' if len(res) > 1 else None for res in pose_shared_helix_resn]
        ref_shared_helix_resn_str_list = [f'{res[0]}-{res[-1]}' if len(res) > 1 else None for res in ref_shared_helix_resn]

        # Calculate the Helix RMSDs on the longest common helices
        shared_helix_rmsds = utils.steph_calculate_RMSDs_on_beta_only_helix(
            pose, ref_pose, 
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

    # Alignment and RMSD arguments
    parser.add_argument('--reference_structure_path', type=str, required=False,
        default='/wynton/home/kortemme/borr/thesis/local_protein_sequence_design/test_inputs/2lv8_inputs/2lv8_cleaned.pdb',
        help='Path to the reference structure.')
    parser.add_argument('--reference_structure_design_info', type=str, required=False,
        default='/wynton/home/kortemme/borr/thesis/local_protein_sequence_design/test_inputs/2lv8_inputs/2lv8_insertion_points.json',
        help="Path to the reference structure's insertion points file.")
    parser.add_argument('--reference_structure_res_start_idx', type=int, required=False, default=1,
        help="Index on which the reference structure's residue numbers begin.")
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